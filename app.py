import base64
import hashlib
import io
import json
import random
import re
import time
import urllib.request

import pandas as pd
import requests
import streamlit as st
from PIL import Image

GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ==========================================
# STREAMLIT
# ==========================================
st.set_page_config(page_title="Robô Solver de Boggle", page_icon="🧩", layout="wide")
st.title("🧩 Robô Solver de Boggle (Netflix)")
st.caption("Upload da foto → IA lê a grade → DFS acha todas as palavras (PT-BR).")


# ==========================================
# ANTI-COTA: debounce + espaçamento
# ==========================================
def qpm_guard(min_interval: float = 1.5):
    agora = time.time()
    ultimo = st.session_state.get("_last_req_ts", 0.0)
    espera = min_interval - (agora - ultimo)
    if espera > 0:
        time.sleep(espera)
    st.session_state["_last_req_ts"] = time.time()


def debounce_click(intervalo: float = 5.0):
    agora = time.time()
    ultimo = st.session_state.get("_last_click_ts", 0.0)
    if agora - ultimo < intervalo:
        st.warning(f"Aguarde {(intervalo - (agora - ultimo)):.1f}s e tente novamente.")
        st.stop()
    st.session_state["_last_click_ts"] = agora


# ==========================================
# DICIONÁRIO PT-BR (CACHE)
# ==========================================
@st.cache_data(ttl=21600)  # 6h
def carregar_dicionario_pt():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    resposta = urllib.request.urlopen(url, timeout=60)
    palavras_raw = resposta.read().decode("utf-8").splitlines()

    dicionario = set()
    prefixos = set()

    for p in palavras_raw:
        p = p.upper().strip()
        if len(p) >= 2 and "-" not in p and "." not in p:
            dicionario.add(p)
            for i in range(1, len(p) + 1):
                prefixos.add(p[:i])

    return dicionario, prefixos


# ==========================================
# BOGGLE DFS
# ==========================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0
    achadas = {}

    direcoes = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def dfs(r, c, visitados, palavra, caminho):
        letra = str(matriz[r][c]).upper().strip()
        nova = palavra + letra

        if nova not in prefixos:
            return

        if nova in dicionario and len(nova) >= 2 and nova not in achadas:
            achadas[nova] = list(caminho)

        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if 0 <= nr < linhas and 0 <= nc < colunas and (nr, nc) not in visitados:
                visitados.add((nr, nc))
                dfs(nr, nc, visitados, nova, caminho + [(nr, nc)])
                visitados.remove((nr, nc))

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return achadas


# ==========================================
# IMAGEM -> B64 + HASH (cache por foto)
# ==========================================
def b64_e_hash(imagem: Image.Image):
    img = imagem.copy().convert("RGB")
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("utf-8"), hashlib.sha256(raw).hexdigest()


# ==========================================
# LISTAR MODELOS (CACHE)
# ==========================================
@st.cache_data(ttl=21600)
def listar_modelos(api_key: str):
    url = f"{GOOGLE_API_BASE}/models?key={api_key}"
    res = requests.get(url, timeout=60)
    res.raise_for_status()
    data = res.json()
    return [
        m["name"]
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def escolher_modelo_image(modelos):
    # pega só os que tem "-image" em qualquer posição aceitável
    modelos_image = [n for n in modelos if re.search(r"(^|-)image($|[-])", n)]
    preferencias = [
        "models/gemini-3.1-flash-image",
        "models/gemini-3.1-flash-lite-image",
        "models/gemini-3-pro-image",
        "models/gemini-2.5-flash-image",
        "models/gemini-3.1-flash-image-preview",
        "models/gemini-3-pro-image-preview",
    ]
    for p in preferencias:
        if p in modelos_image:
            return p
    return modelos_image[0] if modelos_image else None


def extrair_texto_parts(res_json: dict) -> str:
    # Junta todos os parts textuais para evitar KeyError em formatos diferentes
    try:
        cand = res_json.get("candidates", [])[0]
        parts = cand.get("content", {}).get("parts", [])
        textos = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                textos.append(part["text"])
        return "\n".join(textos).strip()
    except Exception:
        return ""


def post_com_retry(url: str, payload: dict, min_interval: float = 1.5, max_retries: int = 6):
    last_resp = None

    for attempt in range(max_retries):
        qpm_guard(min_interval)

        try:
            resp = requests.post(url, json=payload, timeout=120)
            last_resp = resp
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Erro de conexão com a API: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 1.0))
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_s = int(retry_after)
            else:
                wait_s = (2 ** attempt) + random.uniform(0, 1.5)
            st.info(f"API limitou/instável ({resp.status_code}). Retentando em {wait_s:.1f}s...")
            time.sleep(wait_s)
            continue

        # outros erros: não insiste muito
        break

    if last_resp is None:
        raise ValueError("Falha: sem resposta da API.")
    raise ValueError(f"Erro ({last_resp.status_code}): {last_resp.text}")


def extrair_matriz_imagem(imagem: Image.Image, api_key: str, min_interval: float = 1.5):
    if not api_key:
        raise ValueError("Informe a Gemini API Key.")

    # Cache por imagem dentro da sessão
    if "_matriz_cache" not in st.session_state:
        st.session_state["_matriz_cache"] = {}

    img_b64, img_hash = b64_e_hash(imagem)
    if img_hash in st.session_state["_matriz_cache"]:
        return st.session_state["_matriz_cache"][img_hash]

    modelos = listar_modelos(api_key)
    modelo = escolher_modelo_image(modelos)
    if not modelo:
        raise ValueError("Nenhum modelo de visão (-image) disponível nessa chave.")

    url = f"{GOOGLE_API_BASE}/{modelo}:generateContent?key={api_key}"

    prompt = (
        'Analise a imagem do Boggle e extraia APENAS a matriz de letras.\n'
        'ATENÇÃO: algumas células podem ter duas letras juntas (ex.: "CH", "QU"). '
        'Retorne essas duas letras na MESMA string da célula.\n\n'
        'Responda EXCLUSIVAMENTE em JSON estrito, sem markdown, assim:\n'
        '{"matriz":[["E","J","S","Z"],["C","CH","I","F"],["A","B","O","S"],["R","T","U","L"]]}'
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 128,  # curto = menos tokens/min
            "temperature": 0.2
        },
    }

    st.info(f"Modelo em uso: `{modelo}`")
    resp = post_com_retry(url, payload, min_interval=min_interval, max_retries=6)
    data = resp.json()

    texto = extrair_texto_parts(data)
    if not texto:
        raise ValueError(f"Resposta inesperada da API (sem texto): {json.dumps(data)[:1200]}")

    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        raise ValueError("Não veio JSON na resposta. Conteúdo recebido:\n" + texto[:800])

    obj = json.loads(m.group(0))
    matriz = obj.get("matriz")

    if not isinstance(matriz, list) or not matriz or not all(isinstance(l, list) for l in matriz):
        raise ValueError("JSON inválido: 'matriz' não está no formato esperado.")

    # Normaliza e valida formato retangular
    matriz_norm = [[str(cell).upper().strip() for cell in row] for row in matriz]
    larg = len(matriz_norm[0])
    if any(len(row) != larg for row in matriz_norm):
        raise ValueError("Matriz retornada não é retangular (linhas com tamanhos diferentes).")

    st.session_state["_matriz_cache"][img_hash] = matriz_norm
    return matriz_norm


# ==========================================
# UI
# ==========================================
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    tam_minimo = st.number_input("Tamanho mínimo da palavra", min_value=2, max_value=20, value=3, step=1)
    max_listar = st.slider("Máx. palavras exibidas", 50, 3000, 300, 50)
    min_interval = st.slider("Intervalo mínimo entre chamadas (s)", 0.5, 5.0, 1.5, 0.5)


uploaded = st.file_uploader("📷 Envie a foto da grade (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    imagem = Image.open(uploaded)
    st.image(imagem, caption="Imagem enviada", use_container_width=True)

    col1, col2 = st.columns([1, 2], vertical_alignment="center")
    with col1:
        if st.button("🚀 Resolver agora", type="primary"):
            debounce_click(5)

            try:
                with st.spinner("Lendo a grade com a IA..."):
                    matriz = extrair_matriz_imagem(imagem, api_key, min_interval=min_interval)

                st.subheader("Matriz extraída")
                st.dataframe(pd.DataFrame(matriz), use_container_width=True)

                with st.spinner("Buscando palavras (DFS)..."):
                    dicionario, prefixos = carregar_dicionario_pt()
                    achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)

                # filtra e ordena
                palavras = [p for p in achadas.keys() if len(p) >= int(tam_minimo)]
                palavras.sort(key=lambda x: (-len(x), x))

                st.subheader(f"Palavras encontradas (>= {tam_minimo})")
                st.write(f"Total: **{len(palavras)}**")

                df = pd.DataFrame(
                    [{"palavra": p, "tamanho": len(p), "caminho": achadas[p]} for p in palavras]
                )

                st.dataframe(df.head(int(max_listar)), use_container_width=True)

                st.download_button(
                    "⬇️ Baixar CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="boggle_palavras.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")

    with col2:
        st.markdown(
            """
**Pra não estourar cota sem precisar:**
- Não clique várias vezes (tem debounce de 5s).
- A mesma foto fica em cache (não chama API de novo).
- Modelos ficam em cache 6h.
- O slider de intervalo controla o “ritmo” das chamadas.
"""
        )
else:
    st.info("Envie uma imagem para começar.")
