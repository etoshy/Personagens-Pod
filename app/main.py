"""Personagens no pod — API (Flask) + 1 worker na mesma GPU.

POST /jobs                    pedido (JSON) + arquivos (referências, a mestre, bases das "_boca") → gera em fila
GET  /jobs/{id}               estado de cada pose
GET  /jobs/{id}/arquivos/{n}  o PNG final (?qual=bruto: a imagem antes do recorte)
GET  /jobs/{id}/zip           tudo que ficou pronto
POST /jobs/{id}/refazer       gera de novo as poses escolhidas (seed nova, instrução extra opcional)
POST /jobs/{id}/cancelar
GET  /status                  modelo, GPU, fila
"""
from __future__ import annotations

import io
import json
import os
import random
import threading
import time
import traceback
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_file
from PIL import Image

from . import fila, gerador

API_KEY = os.getenv("API_KEY", "")
PRELOAD = os.getenv("PRELOAD", "1") == "1"
PADRAO_GERACAO = {"largura": 1024, "altura": 1536, "passos": int(os.getenv("PASSOS", "40")),
                  "cfg": float(os.getenv("CFG", "4.0"))}
MAX_IMAGENS = 3  # o Qwen-Image-Edit-2511 trabalha bem com até 3 imagens de entrada

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


class Erro(Exception):
    def __init__(self, codigo: int, msg: str):
        super().__init__(msg)
        self.codigo, self.msg = codigo, msg


@app.errorhandler(Erro)
def _erro(e: Erro):
    return jsonify({"detail": e.msg}), e.codigo


def protegido(fn):
    @wraps(fn)
    def _f(*a, **k):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            raise Erro(401, "chave errada")
        return fn(*a, **k)
    return _f


def _abrir(caminho: Path, cor_fundo=(255, 255, 255)) -> Image.Image:
    from PIL import ImageOps
    im = ImageOps.exif_transpose(Image.open(caminho))
    if im.mode in ("RGBA", "LA", "P"):  # referência transparente (a mestre, a base da _boca): põe no fundo da geração
        im = im.convert("RGBA")
        fundo = Image.new("RGB", im.size, tuple(cor_fundo))
        fundo.paste(im, mask=im.split()[3])
        im = fundo
    im = im.convert("RGB")
    im.thumbnail((1536, 1536))
    return im


# ------------------------------------------------------------------ worker
def _processar(jid: str) -> None:
    d = fila.pasta(jid)
    ped = fila.pedido(jid)
    g = dict(PADRAO_GERACAO, **(ped.get("geracao") or {}))
    canvas = tuple(ped.get("canvas") or (1200, 1800))
    refs = [d / "refs" / Path(n).name for n in ped.get("referencias") or []]
    mestre = d / "refs" / Path(ped["mestre"]).name if ped.get("mestre") else None
    poses = {p["arquivo"]: p for p in ped["poses"]}
    ordem = sorted(poses, key=lambda a: (poses[a].get("tipo") == "boca", list(poses).index(a)))
    t0 = time.time()
    while True:  # "refazer" pode devolver poses pra fila enquanto este job ainda roda: dá outra volta
        pend = [a for a in ordem if (fila.status(jid)["itens"].get(a) or {}).get("estado") == "pendente"]
        if not pend:
            break
        _uma_volta(jid, d, ped, g, canvas, refs, mestre, poses, pend)
    st = fila.status(jid)
    if st["status"] == "cancelado":
        return
    erros = sum(1 for x in st["itens"].values() if x.get("estado") == "erro")
    fila.mudar(jid, status="concluido_com_erros" if erros else "concluido", fim=fila.agora(),
               mensagem=f"{len(ordem) - erros} pronta(s) em {time.time() - t0:.0f}s" + (f", {erros} com erro" if erros else ""))


def _uma_volta(jid, d, ped, g, canvas, refs, mestre, poses, ordem) -> None:
    for arquivo in ordem:
        st = fila.status(jid)
        if st["status"] == "cancelado":
            return
        it = st["itens"].get(arquivo) or {}
        if it.get("estado") != "pendente":
            continue
        p = poses[arquivo]
        fila.mudar_item(jid, arquivo, estado="gerando", erro=None)
        fila.mudar(jid, mensagem=f"gerando {arquivo}")
        try:
            t = time.time()
            cor = (0, 255, 0) if ped.get("fundo") == "verde" else (255, 255, 255)
            if p.get("tipo") == "boca":
                base = d / "bruto" / Path(p["base"]).name
                if not base.exists():
                    base = d / "refs" / Path(p["base"]).name
                if not base.exists():
                    raise ValueError(f"falta a imagem base {p['base']} (gere/aprove ela antes)")
                imagens = [_abrir(base, cor)]
            else:
                imagens = [_abrir(r, cor) for r in refs] + ([_abrir(mestre, cor)] if mestre else [])
                imagens = imagens[-MAX_IMAGENS:] if len(imagens) > MAX_IMAGENS else imagens
            prompt = gerador.montar_prompt(p, ped, len(refs), bool(mestre))
            extra = it.get("extra") or p.get("extra")  # correção pedida no refazer (no job ou na pose)
            if extra:
                prompt += f"\nCORRECTION FOR THIS ATTEMPT: {extra}"
            seed = int(it.get("seed") or random.randint(1, 2**31 - 1))
            img = gerador.gerar(prompt, imagens, g["largura"], g["altura"], g["passos"], g["cfg"], seed)
            img.save(d / "bruto" / arquivo)
            info: Dict[str, Any] = {}
            if ped.get("recortar", True):
                alfa = gerador.mascara(img)
                limpeza: Dict[str, Any] = {}
                if ped.get("limpeza", True) is not False:  # 🧹 ilhas, furos, borda, halo branco
                    opl = dict(ped.get("limpeza") if isinstance(ped.get("limpeza"), dict) else {})
                    opl.setdefault("fundo", ped.get("fundo") or "branco")
                    img_l, alfa, limpeza = gerador.limpar(img, alfa, opl)
                else:
                    img_l = img
                final, info = gerador.alinhar(img_l, alfa, canvas, float(ped.get("altura_personagem") or 0.95))
                info["limpeza"] = limpeza
                final.save(d / "saida" / arquivo, optimize=True)
            else:
                img.save(d / "saida" / arquivo)
            fila.mudar_item(jid, arquivo, estado="pronto", seed=seed, segundos=round(time.time() - t, 1),
                            versao=int(it.get("versao") or 0) + 1, aviso=info.get("aviso"), prompt=prompt[:4000],
                            limpeza=info.get("limpeza"),
                            quando=fila.agora())
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            fila.mudar_item(jid, arquivo, estado="erro", erro=f"{type(exc).__name__}: {exc}"[:500])


def _worker() -> None:
    fila.garantir()
    fila.reenfileirar_rodando()
    if PRELOAD:
        try:
            gerador.carregar()
        except Exception:  # noqa: BLE001
            gerador.log("falhou ao carregar — tento de novo no 1º pedido")
    while True:
        jid = fila.proximo()
        if not jid:
            time.sleep(1)
            continue
        try:
            _processar(jid)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            fila.mudar(jid, status="erro", fim=fila.agora(), mensagem=str(exc)[:500])


_WORKER: Dict[str, Any] = {"t": None}


def ligar_worker() -> None:
    if _WORKER["t"] is None or not _WORKER["t"].is_alive():
        _WORKER["t"] = threading.Thread(target=_worker, daemon=True, name="worker")
        _WORKER["t"].start()


# ------------------------------------------------------------------ rotas
@app.get("/")
def inicio():
    return jsonify({"ok": True, "servico": "personagens-pod", "rotas": ["/status", "/jobs"]})


@app.get("/status")
@protegido
def status_geral():
    gpu = None
    if not gerador.MOCK:
        try:
            import torch
            if torch.cuda.is_available():
                livre, total = torch.cuda.mem_get_info()
                gpu = {"nome": torch.cuda.get_device_name(0), "total_gb": round(total / 1e9, 1), "livre_gb": round(livre / 1e9, 1)}
        except Exception:  # noqa: BLE001
            pass
    jobs = fila.listar()
    return jsonify({"modelo": gerador.ESTADO, "mock": gerador.MOCK, "gpu": gpu, "modelo_id": gerador.MODELO,
                    "fila": sum(1 for j in jobs if j["status"] in ("na_fila", "rodando")), "padrao": PADRAO_GERACAO})


@app.post("/jobs")
@protegido
def novo_job():
    try:
        ped = json.loads(request.form.get("pedido") or "")
    except ValueError:
        raise Erro(400, "pedido não é JSON")
    poses = [p for p in ped.get("poses") or [] if isinstance(p, dict) and p.get("arquivo") and p.get("prompt")]
    if not poses:
        raise Erro(400, "nenhuma pose (cada uma precisa de \"arquivo\" e \"prompt\")")
    nomes = set()
    for p in poses:
        p["arquivo"] = Path(p["arquivo"]).name
        if not p["arquivo"].lower().endswith(".png"):
            p["arquivo"] += ".png"
        if p["arquivo"] in nomes:
            raise Erro(400, f"arquivo repetido: {p['arquivo']}")
        nomes.add(p["arquivo"])
    ped["poses"] = poses
    dados = {Path(a.filename).name: a.read() for a in request.files.getlist("arquivos") if a.filename}
    for n in list(ped.get("referencias") or []) + ([ped["mestre"]] if ped.get("mestre") else []):
        if Path(n).name not in dados:
            raise Erro(400, f"faltou mandar o arquivo {n}")
    if any(p.get("tipo") != "boca" for p in poses) and not ped.get("referencias") and not ped.get("mestre"):
        raise Erro(400, "mande pelo menos uma imagem de referência da personagem")
    jid = fila.criar(ped, dados)
    ligar_worker()
    return jsonify({"id": jid, "poses": len(poses), "posicao": fila.posicao(jid)})


@app.get("/jobs")
@protegido
def listar_jobs():
    return jsonify([{k: j.get(k) for k in ("id", "nome", "status", "criado", "fim", "progresso", "mensagem")} for j in fila.listar()])


def _pasta(jid: str) -> Path:
    try:
        d = fila.pasta(jid)
    except KeyError:
        raise Erro(404, "job não existe")
    if not (d / "status.json").exists():
        raise Erro(404, "job não existe")
    return d


@app.get("/jobs/<jid>")
@protegido
def ver_job(jid: str):
    _pasta(jid)
    return jsonify(dict(fila.status(jid), posicao=fila.posicao(jid)))


@app.get("/jobs/<jid>/arquivos/<nome>")
@protegido
def baixar(jid: str, nome: str):
    d = _pasta(jid)
    cam = d / ("bruto" if request.args.get("qual") == "bruto" else "saida") / Path(nome).name
    if not cam.exists():
        raise Erro(404, "ainda não existe")
    return send_file(cam, mimetype="image/png", max_age=0)


@app.get("/jobs/<jid>/zip")
@protegido
def zip_job(jid: str):
    d = _pasta(jid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((d / "saida").glob("*.png")):
            z.write(f, f.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{jid}.zip")


@app.post("/jobs/<jid>/refazer")
@protegido
def refazer(jid: str):
    _pasta(jid)
    st = fila.status(jid)
    corpo = request.get_json(silent=True) or {}
    alvo = [a for a in corpo.get("arquivos") or [] if a in st["itens"]]
    if not alvo:
        raise Erro(400, "diga quais arquivos refazer")
    extra = str(corpo.get("extra") or "")[:600]
    for a in alvo:
        fila.mudar_item(jid, a, estado="pendente", erro=None, seed=None, extra=extra or None)
    if st["status"] in fila.FINAIS:
        fila.mudar(jid, status="na_fila", mensagem=f"refazendo {len(alvo)}")
    ligar_worker()
    return jsonify({"ok": True, "refazendo": alvo})


@app.post("/jobs/<jid>/cancelar")
@protegido
def cancelar(jid: str):
    _pasta(jid)
    if fila.status(jid)["status"] not in fila.FINAIS:
        fila.mudar(jid, status="cancelado", fim=fila.agora(), mensagem="cancelado")
    return jsonify({"ok": True})


def main() -> None:
    ligar_worker()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), threaded=True)


if __name__ == "__main__":
    main()
