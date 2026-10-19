"""Fila em disco: cada pedido vira jobs/<id>/ (pedido.json, status.json, refs/, bruto/, saida/)."""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/personagens_data"))
JOBS = DATA_DIR / "jobs"
FINAIS = ("concluido", "concluido_com_erros", "erro", "cancelado")
_TRAVA = threading.RLock()


def agora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def garantir() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)


def pasta(jid: str) -> Path:
    if not jid or not all(c.isalnum() or c in "-_" for c in jid):
        raise KeyError(jid)
    return JOBS / jid


def ler(caminho: Path, padrao: Any = None) -> Any:
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return padrao


def gravar(caminho: Path, dados: Any) -> None:
    tmp = caminho.with_suffix(caminho.suffix + f".{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, caminho)


def criar(pedido: Dict[str, Any], arquivos: Dict[str, bytes]) -> str:
    garantir()
    jid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = pasta(jid)
    for sub in ("refs", "bruto", "saida"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for nome, dados in arquivos.items():
        (d / "refs" / Path(nome).name).write_bytes(dados)
    gravar(d / "pedido.json", pedido)
    itens = {p["arquivo"]: {"estado": "pendente", "erro": None, "versao": 0} for p in pedido["poses"]}
    gravar(d / "status.json", {"id": jid, "nome": pedido.get("nome") or "", "status": "na_fila", "criado": agora(),
                               "inicio": None, "fim": None, "mensagem": "na fila", "itens": itens,
                               "progresso": {"feito": 0, "total": len(itens)}})
    return jid


def status(jid: str) -> Dict[str, Any]:
    st = ler(pasta(jid) / "status.json")
    if not st:
        raise KeyError(jid)
    return st


def pedido(jid: str) -> Dict[str, Any]:
    return ler(pasta(jid) / "pedido.json", {})


def mudar(jid: str, **campos: Any) -> Dict[str, Any]:
    with _TRAVA:
        st = status(jid)
        st.update(campos)
        gravar(pasta(jid) / "status.json", st)
        return st


def mudar_item(jid: str, arquivo: str, **campos: Any) -> Dict[str, Any]:
    with _TRAVA:
        st = status(jid)
        st["itens"].setdefault(arquivo, {}).update(campos)
        feitos = sum(1 for x in st["itens"].values() if x.get("estado") in ("pronto", "erro"))
        st["progresso"] = {"feito": feitos, "total": len(st["itens"])}
        gravar(pasta(jid) / "status.json", st)
        return st


def listar() -> List[Dict[str, Any]]:
    garantir()
    out = []
    for d in sorted(JOBS.iterdir()):
        st = ler(d / "status.json")
        if st:
            out.append(st)
    return out


def proximo() -> Optional[str]:
    with _TRAVA:
        for st in listar():
            if st["status"] == "na_fila":
                mudar(st["id"], status="rodando", inicio=st.get("inicio") or agora(), mensagem="começando")
                return st["id"]
    return None


def posicao(jid: str) -> int:
    fila = [s["id"] for s in listar() if s["status"] in ("na_fila", "rodando")]
    return fila.index(jid) + 1 if jid in fila else 0


def reenfileirar_rodando() -> None:
    """O processo caiu no meio: o que estava rodando volta pra fila (o que já ficou pronto continua pronto)."""
    for st in listar():
        if st["status"] == "rodando":
            for a, it in st["itens"].items():
                if it.get("estado") == "gerando":
                    mudar_item(st["id"], a, estado="pendente")
            mudar(st["id"], status="na_fila", mensagem="reenfileirado após reinício")
