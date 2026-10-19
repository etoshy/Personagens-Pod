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
QUANT = os.getenv("QUANT", "")                  # "" (bf16) | nf4 | int8 — modelo comprimido: cabe INTEIRO numa placa de 48 GB

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
            kw: Dict[str, Any] = {}
            if QUANT in ("nf4", "int8"):
                # 🗜 o "transformer" (~40 GB) comprimido: nf4 ≈ 11 GB, int8 ≈ 20 GB → o modelo inteiro fica na GPU
                # (sem dividir com a RAM: mais rápido e sem o processo morrer por falta de RAM)
                from diffusers import PipelineQuantizationConfig
                if QUANT == "nf4":
                    qk = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16}
                    qb = "bitsandbytes_4bit"
                else:
                    qk, qb = {"load_in_8bit": True}, "bitsandbytes_8bit"
                kw["quantization_config"] = PipelineQuantizationConfig(quant_backend=qb, quant_kwargs=qk,
                                                                       components_to_quantize=["transformer"])
                log(f"modelo comprimido: {QUANT}")
            _pipe = QwenImageEditPlusPipeline.from_pretrained(MODELO, torch_dtype=torch.bfloat16, **kw)
            if LIGHTNING_REPO:
                log(f"LoRA lightning: {LIGHTNING_REPO} {LIGHTNING_FILE}")
                _pipe.load_lora_weights(LIGHTNING_REPO, **({"weight_name": LIGHTNING_FILE} if LIGHTNING_FILE else {}))
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            usar_offload = OFFLOAD == "1" or (OFFLOAD == "auto" and vram < 70 and not QUANT)
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
    im = Image.new("RGB", (largura, altura), (0, 255, 0) if "GREEN" in prompt.upper() else (255, 255, 255))
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
        import numpy as np
        c = np.asarray(img.convert("RGB")).astype(int)
        fundo = (c.min(axis=2) >= 245) | ((c[..., 1] > 200) & (c[..., 0] < 60) & (c[..., 2] < 60))
        return Image.fromarray(((~fundo) * 255).astype("uint8"), "L").filter(ImageFilter.MedianFilter(3))
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(img.convert("RGB")).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        pred = _birefnet(x)[-1].sigmoid().cpu()[0].squeeze()
    return transforms.ToPILImage()(pred.float()).resize(img.size, Image.LANCZOS)


PADRAO_LIMPEZA = {"fundo": "branco", "ilhas": True, "ilha_min": 0.02, "furos_px": 400, "fundo_preso": True, "branco_min": 248, "fundo_preso_px": 600,
                  "erodir": 1, "halo": True, "suavizar": 0.6}


def limpar(img: Image.Image, alfa: Image.Image, opcoes: Optional[Dict[str, Any]] = None) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """🧹 Limpa o recorte. fundo "verde" (recomendado: chroma key — o verde preso entre os fios do cabelo sai também,
    e o reflexo verde da borda é neutralizado) ou "branco" (os passos 3 e 5 abaixo):
      1. ilhas  — some com os pontinhos soltos (pedaços < ilha_min do maior pedaço); objeto separado grande
                  (ex.: o buraco negro flutuando sobre a mão) fica;
      2. furos  — fecha os furinhos DENTRO da personagem (< furos_px pixels) que o recorte deixou; vão de verdade
                  (entre o braço e o corpo) é grande e continua transparente;
      3. fundo_preso — o branco PURO do fundo que o recorte marcou como personagem vira transparente: o que
                  encosta no lado de fora (entre as pernas, em volta do cabelo e de um objeto, qualquer tamanho) e mancha
                  grande fechada (≥ fundo_preso_px). Só branco puro e sem cor (média ≥ 242) — camisa marfim, babado e
                  o brilho do olho ficam;
      4. erodir — come N px da borda (onde o branco do fundo ainda gruda);
      5. halo   — "descontamina" a borda semitransparente: tira o branco que vazou na cor (cabelo sem auréola);
      6. suavizar — desfoca 0,6 px só o alfa (borda sem serrilhado).
    → (imagem RGB limpa, alfa limpo, o que fez)."""
    import numpy as np
    o = dict(PADRAO_LIMPEZA, **(opcoes or {}))
    a = np.asarray(alfa.convert("L"), dtype=np.float32) / 255.0
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    feito: Dict[str, Any] = {"ilhas_removidas": 0, "furos_fechados": 0}
    verde = o.get("fundo") == "verde"
    if verde:  # 🟩 chroma key: TODO verde do fundo sai — inclusive o preso entre os fios do cabelo
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        outro = np.maximum(r, b)
        chave = (g >= 90) & (g > outro * 1.35 + 20)
        a[chave] = 0.0
        feito["verde_px"] = int(chave.sum())
        borda_verde = (a > 0) & (g > outro + 8)
        excesso = np.clip((g - outro) / 160.0, 0, 1)
        a[borda_verde] *= 1.0 - excesso[borda_verde]          # borda esverdeada fica mais transparente
        rgb[..., 1] = np.where(g > outro, outro + (g - outro) * 0.15, g)  # tira o reflexo verde (despill)
    try:
        import cv2
    except ImportError:  # sem OpenCV: só o halo e o limiar
        cv2 = None
    if cv2 is not None and o.get("ilhas"):
        bin_ = (a > 0.5).astype(np.uint8)
        n, lab, stats, _c = cv2.connectedComponentsWithStats(bin_, connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            maior = int(areas.max())
            for k, ar in enumerate(areas, start=1):
                if ar < maior * float(o["ilha_min"]):
                    regiao = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    a[regiao & (bin_ == 0) | (lab == k)] = 0.0
                    feito["ilhas_removidas"] += 1
    if cv2 is not None and int(o.get("furos_px") or 0) > 0:
        fundo = (a <= 0.5).astype(np.uint8)
        n, lab, stats, _c = cv2.connectedComponentsWithStats(fundo, connectivity=4)
        h, w = a.shape
        for k in range(1, n):
            x, y, bw, bh, ar = stats[k]
            toca_borda = x == 0 or y == 0 or x + bw >= w or y + bh >= h
            if not toca_borda and ar < int(o["furos_px"]):
                a[lab == k] = 1.0
                feito["furos_fechados"] += 1
    if verde:  # o "fechar furinhos" não pode devolver o verde que a chave tirou
        a[chave] = 0.0
    if cv2 is not None and o.get("fundo_preso") and not verde:
        lim = float(o.get("branco_min") or 248)
        mn = rgb.min(axis=2)
        claro = ((mn >= lim - 13) & ((rgb.max(axis=2) - mn) <= 8) & (a > 0.5)).astype(np.uint8)
        n, lab, stats, _c = cv2.connectedComponentsWithStats(claro, connectivity=8)
        if n > 1:
            fora = cv2.dilate((a <= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            toca = np.bincount(lab[fora & (claro > 0)].ravel(), minlength=n) > 0      # o pedaço encosta no lado de fora
            media = np.bincount(lab.ravel(), weights=mn.ravel(), minlength=n) / np.maximum(1, stats[:, cv2.CC_STAT_AREA])
            area = stats[:, cv2.CC_STAT_AREA]
            sai = np.zeros(n, bool)
            sai[1:] = (toca[1:] & (media[1:] >= lim - 6)) | ((area[1:] >= int(o.get("fundo_preso_px") or 600)) & (media[1:] >= lim - 2))
            tirar = sai[lab]
            if tirar.any():
                tirar = cv2.dilate(tirar.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # a borda clarinha junto
                quase = (mn >= lim - 30) & ((rgb.max(axis=2) - mn) <= 14)
                a[tirar & quase] = 0.0
                feito["fundo_preso_px"] = int((tirar & quase).sum())
    if cv2 is not None and int(o.get("erodir") or 0) > 0:
        r = int(o["erodir"])
        a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    a[a < 0.04] = 0.0
    a[a > 0.96] = 1.0
    if o.get("halo") and not verde:
        borda = (a > 0.0) & (a < 1.0)
        aa = np.clip(a[borda], 0.05, 1.0)[:, None]
        rgb[borda] = np.clip((rgb[borda] - (1.0 - aa) * 255.0) / aa, 0, 255)  # C = a·F + (1−a)·branco → F
        feito["borda_px"] = int(borda.sum())
    alfa_out = Image.fromarray((a * 255).round().astype(np.uint8), "L")
    if float(o.get("suavizar") or 0) > 0:
        alfa_out = alfa_out.filter(ImageFilter.GaussianBlur(float(o["suavizar"])))
    return Image.fromarray(rgb.round().astype(np.uint8), "RGB"), alfa_out, feito


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
