# Personagens no pod — poses PNG da mesma personagem (prova de conceito)

Gera as poses de uma personagem do canal (reações, segurando objetos, "_boca" pra fala) a partir da ficha dela,
no seu pod da RunPod:

- 🎨 **Qwen-Image-Edit-2511** (Apache 2.0) — mesma personagem, pose nova (até 3 imagens de entrada: a ficha +
  a imagem **mestre** aprovada);
- ✂️ **BiRefNet** (MIT) — tira o fundo;
- 🧹 **limpeza** depois do recorte — a geração sai em **fundo verde (chroma key)** e todo verde sai, inclusive o
  preso entre os fios do cabelo, entre o braço e o corpo e em volta de objetos; o reflexo verde da borda é
  neutralizado; pontinhos soltos somem, furinhos dentro do corpo fecham, a borda perde 1 px e fica suave.
  (Com `"fundo": "branco"` a limpeza tira o branco puro preso e o halo branco — camisa marfim e olho ficam.);
- 📐 cada PNG sai **1200 × 1800, transparente**, com a personagem a ~95% da altura e encostada embaixo (o corte do
  quadril é a borda) — a cabeça fica no mesmo lugar em todas as poses.

Quem usa é o **BestEditor** (🧍 Personagens no pod): ele manda as referências e o JSON de poses, mostra a galeria
e você aprova/refaz. A API também funciona sozinha (`/jobs`).

---

## 1. Criar o pod (uma vez)

- **GPU**: o ideal é **80 GB** (A100 80GB / H100) — o modelo (~20B) + o leitor de texto (~7B) cabem inteiros e cada
  imagem leva ~30–60 s. Na **L40S 48 GB** também roda: o `OFFLOAD=auto` divide com a RAM (mais lento; precisa de
  ~64 GB de RAM no pod). ⚠ Evite RTX 4090 24 GB pra este modelo.
- **Template**: RunPod PyTorch (2.4 ou mais novo).
- **Network Volume** de **~120 GB** em `/workspace` (o modelo tem ~60 GB e fica no volume pras próximas sessões).
- **Expose HTTP Ports**: `8000`.

## 2. Instalar (uma vez por volume)

```bash
cd /workspace
git clone https://github.com/<você>/personagens-pod.git   # ou envie a pasta pelo Jupyter
bash personagens-pod/setup.sh        # cria o ambiente e baixa os modelos (~60 GB, 10–30 min na 1ª vez)
```

## 3. Cada sessão

```bash
cd /workspace/personagens-pod && git pull && bash setup.sh && bash start.sh
```

Mostra a URL `https://<id-do-pod>-8000.proxy.runpod.net` — cole no editor (🧍 Personagens no pod).
O modelo carrega em segundo plano (1–3 min com ele já no volume). Acompanhe: `tail -f /workspace/personagens_logs/api.log`
ou `GET /status` (`"modelo": "pronto"`). Se o processo cair, ele volta sozinho em 5 s.

| Variável | Efeito |
|---|---|
| `PORT=8001` | outra porta (ex.: no mesmo pod do Studio IA — mas os dois na mesma GPU não cabem) |
| `API_KEY=segredo` | passa a exigir a chave no header `X-API-Key` |
| `PASSOS=40` `CFG=4.0` | padrão de geração (o editor pode mudar por pedido) |
| `OFFLOAD=auto\|1\|0` | divide o modelo com a RAM (auto = placa < 70 GB) |
| `RAPIDO=1` | ⚡ LoRA [Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning): 4 passos e CFG 1 — ~10× mais rápido (os passos do pedido são ignorados). Qualidade um pouco menor: compare com a mestre |
| `LIGHTNING_REPO=…` `LIGHTNING_FILE=…` | outro LoRA Lightning (o `RAPIDO=1` já preenche os dois) |
| `MOCK=1` | sem modelo: desenha um boneco (testar o caminho todo sem GPU) |

No fim: **desligue o pod** (Stop mantém o volume).

## 4. Como funciona (o fluxo recomendado)

1. **Mestre**: gera só a pose principal (ex.: `vanta_neutra_esq`) com a ficha como referência. Refaça até ficar
   igual à ficha e **aprove como mestre**.
2. **O resto**: cada pose vai com a ficha **+ a mestre** → a personagem fica igual.
3. **`_boca`**: edição da imagem pronta (mesma imagem, só a boca aberta) — pro editor alternar enquanto ela fala.
4. Você aprova/refaz cada uma (refazer aceita uma correção: "o olho de eclipse é do outro lado", "sem cigarro"…).

Próximo passo (depois da prova de conceito): com ~20 aprovadas, treinar um **LoRA** da personagem no mesmo pod
(a consistência sobe muito e as poses novas saem só pelo texto).

## API

`POST /jobs` (multipart): campo `pedido` (JSON) + `arquivos` (as imagens citadas no pedido).

```json
{
  "nome": "vanta lote 1",
  "referencias": ["vanta_ficha.png"],
  "descricoes": {"vanta_ficha.png": "the official character reference sheet of Vanta"},
  "mestre": "vanta_neutra_esq.png",
  "texto_fixo": {"trava": "VANTA — CHARACTER LOCK …", "estilo": "ART STYLE …", "regras": "TECHNICAL RULES …"},
  "geracao": {"largura": 1024, "altura": 1536, "passos": 40, "cfg": 4.0},
  "canvas": [1200, 1800], "altura_personagem": 0.95, "recortar": true,
  "fundo": "verde", "limpeza": {"ilhas": true, "furos_px": 400, "erodir": 1, "suavizar": 0.6},
  "poses": [
    {"arquivo": "vanta_cetica_esq.png", "tipo": "pose", "prompt": "skeptical, arms crossed …"},
    {"arquivo": "vanta_cetica_esq_boca.png", "tipo": "boca", "base": "vanta_cetica_esq.png", "prompt": "Keep everything identical …"}
  ]
}
```

- `GET /jobs/<id>` — estado de cada pose (`pendente`, `gerando`, `pronto`, `erro`), seed, tempo, versão, aviso.
- `GET /jobs/<id>/arquivos/<nome>` — o PNG final (`?qual=bruto` = antes do recorte).
- `GET /jobs/<id>/zip` · `POST /jobs/<id>/refazer` `{"arquivos": [...], "extra": "correção"}` · `POST /jobs/<id>/cancelar`.
- `GET /status` — modelo, GPU, fila.

Arquivos no pod: `/workspace/personagens_data/jobs/<id>/` (`refs/`, `bruto/`, `saida/`, `status.json`).
