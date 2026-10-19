"""Gera as poses: Qwen-Image-Edit-2511 (mesma personagem, pose nova) → BiRefNet (tira o fundo) → canvas fixo.

MOCK=1 desenha um boneco no lugar do modelo (pra testar o caminho todo sem GPU)."""
from __future__ import annotations

import inspect
import os
import random
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

MOCK = os.getenv("MOCK") == "1"
MODELO = os.getenv("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
RECORTE = os.getenv("RECORTE_MODEL", "ZhengPeng7/BiRefNet")
OFFLOAD = os.getenv("OFFLOAD", "auto")          # auto | 1 | 0
LIGHTNING_REPO = os.getenv("LIGHTNING_REPO", "")  # opcional: um LoRA "lightning" (poucos passos) do 2511
LIGHTNING_FILE = os.getenv("LIGHTNING_FILE", "")

_pipe = None
_birefnet = None
_TRAVA = threading.Lock()
ESTADO: Dict[str, Any] = {"modelo": "nao_carregado", "erro": None, "carregado_em": None, "offload": None}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ carregar
def carregar() -> None:
    global _pipe, _birefnet
    with _TRAVA:
        if _pipe is not None or MOCK:
            ESTADO["modelo"] = "pronto" if (_pipe is not None or MOCK) else ESTADO["modelo"]
            return
        ESTADO.update(modelo="carregando", erro=None)
        t = time.time()
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
            log(f"carregando {MODELO} (a 1ª vez baixa ~60 GB pro volume)…")
            _pipe = QwenImageEditPlusPipeline.from_pretrained(MODELO, torch_dtype=torch.bfloat16)
            if LIGHTNING_REPO:
                log(f"LoRA lightning: {LIGHTNING_REPO} {LIGHTNING_FILE}")
                _pipe.load_lora_weights(LIGHTNING_REPO, **({"weight_name": LIGHTNING_FILE} if LIGHTNING_FILE else {}))
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            usar_offload = OFFLOAD == "1" or (OFFLOAD == "auto" and vram < 70)
            if usar_offload:
                _pipe.enable_model_cpu_offload()  # placa < 70 GB: cada parte vai pra GPU só na hora dela
            else:
                _pipe.to("cuda")
            ESTADO["offload"] = usar_offload
            log(f"Qwen-Image-Edit pronto em {time.time() - t:.0f}s (GPU {vram:.0f} GB, offload={usar_offload})")
            from transformers import AutoModelForImageSegmentation
            _birefnet = AutoModelForImageSegmentation.from_pretrained(RECORTE, trust_remote_code=True)
            _birefnet.to("cuda").eval().half()
            log("BiRefNet pronto")
            ESTADO.update(modelo="pronto", carregado_em=time.strftime("%H:%M:%S"))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _pipe = None
            ESTADO.update(modelo="erro", erro=f"{type(exc).__name__}: {exc}"[:800])
            raise


# ------------------------------------------------------------------ prompt
def montar_prompt(pose: Dict[str, Any], pedido: Dict[str, Any], n_refs: int, tem_mestre: bool) -> str:
    """O texto final de uma pose. As imagens vão na ordem: referências (ficha…) e por último a mestre aprovada."""
    if pose.get("tipo") == "boca":
        return pose["prompt"]
    fixo = pedido.get("texto_fixo") or {}
    desc = pedido.get("descricoes") or {}
    partes = []
    for k, nome in enumerate(pedido.get("referencias") or [], 1):
        partes.append(f"Image {k} is {desc.get(nome) or 'the official character reference sheet'}.")
    if tem_mestre:
        partes.append(f"Image {n_refs + 1} is the APPROVED MASTER image of this character — match its face, hair, eyes, "
                      "outfit, colors, art style, scale and framing exactly.")
    partes.append(f"Draw ONE new image of this same character, alone: {pose['prompt']}")
    for chave in ("trava", "estilo", "regras"):
        if fixo.get(chave):
            partes.append(str(fixo[chave]))
    return "\n".join(partes)


# ------------------------------------------------------------------ gerar
def _mock(prompt: str, imagens: List[Image.Image], largura: int, altura: int, seed: int) -> Image.Image:
    rnd = random.Random(seed)
    im = Image.new("RGB", (largura, altura), (255, 255, 255))
    d = ImageDraw.Draw(im)
    cor = (rnd.randint(20, 90), rnd.randint(20, 60), rnd.randint(60, 140))
    cx = largura // 2 + rnd.randint(-60, 60)
    d.ellipse([cx - 150, 120, cx + 150, 460], fill=(240, 220, 200), outline=(0, 0, 0), width=6)
    d.polygon([(cx - 330, altura), (cx - 200, 470), (cx + 200, 470), (cx + 330, altura)], fill=cor)
    d.text((40, 30), prompt[:90], fill=(0, 0, 0))
    if imagens:
        im.paste(imagens[0].resize((160, 220)), (largura - 180, 20))
    return im


def gerar(prompt: str, imagens: List[Image.Image], largura: int, altura: int, passos: int, cfg: float, seed: int) -> Image.Image:
    if MOCK:
        time.sleep(0.05)
        return _mock(prompt, imagens, largura, altura, seed)
    import torch
    carregar()
    gen = torch.Generator("cuda").manual_seed(int(seed))
    kw: Dict[str, Any] = {"image": imagens, "prompt": prompt, "negative_prompt": " ", "true_cfg_scale": float(cfg),
                          "num_inference_steps": int(passos), "generator": gen}
    params = inspect.signature(_pipe.__call__).parameters
    if "guidance_scale" in params:
        kw["guidance_scale"] = 1.0
    if "height" in params and "width" in params:
        kw.update(height=int(altura), width=int(largura))
    with torch.inference_mode():
        return _pipe(**kw).images[0]


# ------------------------------------------------------------------ recortar e alinhar
def mascara(img: Image.Image) -> Image.Image:
    """Alfa da personagem (L, 0–255)."""
    if MOCK:
        cinza = img.convert("L")
        return cinza.point(lambda v: 255 if v < 245 else 0).filter(ImageFilter.MedianFilter(3))
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(img.convert("RGB")).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        pred = _birefnet(x)[-1].sigmoid().cpu()[0].squeeze()
    return transforms.ToPILImage()(pred.float()).resize(img.size, Image.LANCZOS)


def alinhar(img: Image.Image, alfa: Image.Image, canvas: Tuple[int, int] = (1200, 1800),
            altura_personagem: float = 0.95) -> Tuple[Image.Image, Dict[str, Any]]:
    """PNG transparente no canvas fixo: a personagem com ~95% da altura, encostada na borda de baixo (o corte do
    quadril vira a borda), centrada pelo próprio desenho. Assim todas as poses têm a cabeça no mesmo lugar."""
    rgba = img.convert("RGBA")
    rgba.putalpha(alfa)
    caixa = alfa.point(lambda v: 255 if v > 24 else 0).getbbox()
    W, H = canvas
    if not caixa:
        raise ValueError("o recorte não achou a personagem")
    fig = rgba.crop(caixa)
    k = (H * altura_personagem) / fig.height
    if fig.width * k > W * 0.98:  # braços muito abertos: diminui pra caber na largura
        k = (W * 0.98) / fig.width
    fig = fig.resize((max(1, round(fig.width * k)), max(1, round(fig.height * k))), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(fig, ((W - fig.width) // 2, H - fig.height), fig)
    toca_baixo = caixa[3] >= img.height - 4
    return out, {"escala": round(k, 3), "caixa": list(caixa), "cortada_embaixo": toca_baixo,
                 "aviso": None if toca_baixo else "a personagem não encosta na borda de baixo (corpo inteiro?) — confira o enquadramento"}
