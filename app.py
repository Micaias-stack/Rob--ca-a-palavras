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


# ==========================================
# CONFIG STREAMLIT
# ==========================================
st.set_page_config(page_title="Robô Solver de Boggle", page_icon="🧩", layout="wide")
st.title("🧩 Robô Solver de Boggle (Netflix)")


# ==========================================
# CONTROLES DE “ANTI-ESTOURO DE COTA”
# ==========================================
def qpm_guard(min_interval=1.5):
    """Garante espaçamento mínimo entre chamadas HTTP (evita estourar RPM/QPM)."""
    agora = time.time()
    ultimo = st.session_state.get("_last_req_ts", 0.0)
    espera = min_interval - (agora - ultimo)
    if espera > 0:
        time.sleep(espera)
    st.session_state["_last_req_ts"] = time.time()


def debounce_click(intervalo=5):
    """Evita clique duplo no botão (Streamlit pode rerodar rápido)."""
    agora = time.time()
    ultimo = st.session_state.get("_last_click_ts", 0.0)
    if agora - ultimo < intervalo:
        faltam = intervalo - (agora - ultimo)
        st.warning(f"Aguarde {faltam:.1f}s antes de tentar de novo.")
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
# BUSCA DFS (Boggle)
# ==========================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas > 0 else 0
    palavras_encontradas = {}

    direcoes = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def dfs(r, c, visitados, palavra_atual, caminho):
        letra = str(matriz[r][c]).upper().strip()
        nova = palavra_atual + letra

        if nova not in prefixos:
            return

        if nova in dicionario and len(nova) >= 2:
            if nova not in palavras_encontradas:
                palavras_encontradas[nova] = list(caminho)

        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if 0 <= nr < linhas and 0 <= nc < colunas and (nr, nc) not in visitados:
                visitados.add((nr, nc))
                dfs(nr, nc, visitados, nova, caminho + [(nr, nc)])
                visitados.remove((nr, nc))

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return palavras_encontradas


# ==========================================
# IMAGEM -> BASE64 + HASH (cache por foto)
# ==========================================
def b64_e_hash_da_imagem(imagem: Image.Image):
    img = imagem.copy().convert("RGB")
    img.thumbnail((900, 900))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()

    img_b64 = base64.b64encode(raw).decode("utf-8")
    img_hash = hashlib.sha256(raw).hexdigest()
    return img_b64, img_hash


# ==========================================
# LISTAR MODELOS (CACHE 6h)
# ==========================================
@st.cache_data(ttl=21600)
def listar_modelos(api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    res = requests.get(url, timeout=60)
    res.raise_for_status()
    data = res.json()

    return [
        m["name"]
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


# ==========================================
# EXTRAIR MATRIZ VIA GEMINI (modelo -image, retry 429/503)
# ==========================================
def extrair_matriz_imagem(imagem: Image.Image, api_key: str, min_interval=1.5):
    if not api_key:
        raise ValueError("Informe a Gemini API Key.")

    # Cache por imagem (se a foto for a mesma, não chama API de novo)
    if "_matriz_cache" not in st.session_state:
        st.session_state["_matriz_cache"] = {}

    img_b64, img_hash = b64_e_hash_da_imagem(imagem)
    if img_hash in st.session_state["_matriz_cache"]:
        return st.session_state["_matriz_cache"][img_hash]

    # Listar modelos (cacheado 6h)
    qpm_guard(min_interval)
    modelos_gc = listar_modelos(api_key)

    # Filtra modelos de visão (contendo "-image")
    modelos_image = [n for n in modelos_gc if re.search(r"(^|-)image($|[-])", n)]

    preferencias = [
        "models/gemini-3.1-flash-image",
        "models/gemini-3.1-flash-lite-image",
        "models/gemini-3-pro-image",
        "models/gemini-2.5-flash-image",
        "models/gemini-3.1-flash-image-preview",
        "models/gemini-3-pro-image-preview",
    ]

    modelo = next((p for p in preferencias if p in modelos_image), None)
    if not modelo and modelos_image:
        modelo = modelos_image[0]

    if not modelo:
        raise ValueError(
            "Nenhum modelo de visão (-image) encontrado nesta chave. "
            f"Modelos disponíveis: {modelos_gc}"
        )

    generate_url = f"https://generativelanguage.googleapis.com/v1beta/{modelo}:generateContent?key={api_key}"

    prompt = (
        'Analise esta imagem de um jogo de palavras (Boggle). '
        'Extraia APENAS a matriz/grade completa de letras. '
        'IMPORTANTE: algumas células podem ter DUAS letras juntas (ex.: "CH", "QU", "RR"). '
        'Extraia exatamente como aparece no bloco.\n\n'
        'Responda EXCLUSIVAMENTE em JSON estrito, sem markdown, neste formato:\n'
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
        # resposta curta = menos tokens/min
        "generationConfig": {"maxOutputTokens": 128, "temperature": 0.2},
    }

    max_retries = 6
    response = None

    for attempt in range(max_retries):
        qpm_guard(min_interval)

        try:
            response = requests.post(generate_url, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Erro de conexão com a API: {e}")

        if response.status_code == 200:
            break

        # 429/503: respeitar Retry-After quando existir
        if response.status_code in (429, 503) and attempt < max_retries - 1:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_s = int(retry_after)
            else:
                wait_s = (2 ** attempt) + random.uniform(0, 1.5)
            st.info(f"Limite/instabilidade ({response.status_code}). Retentando em {wait_s:.1f}s...")
            time.sleep(wait_s)
            continue

        break

    if response is None:
        raise ValueError("Falha sem resposta da API.")
    if response.status_code != 200:
        raise ValueError(f"Erro ({response.status_code}): {response.text}")

    data = response.json()

    # junta todos os parts text (às vezes vem em mais de um)
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texto = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
    except Exception:
        raise ValueError(f"Resposta inesperada da API: {json.dumps(data)}")

    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("A IA não retornou JSON válido. Resposta: " + texto)

    obj = json.loads(match.group())
    matriz = obj.get("matriz")

    if not isinstance(matriz, list) or not matriz or not isinstance(matriz[0], list):
        raise ValueError("JSON inválido: chave 'matriz' não está no formato esperado.")

    # normaliza strings
    matriz_norm = [[str(x).upper().strip() for x in linha] for linha in matriz]

    # salva cache por imagem
    st.session_state["_matriz_cache"][img_hash] = matriz_norm
    return matriz_norm


# ==========================================
# UI
# ==========================================
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Gemini API Key", type="password")
    tam_minimo = st.number_input("Tamanho mínimo", min_value=2, max_value=20, value=3, step=1)
    max_listar = st.slider("Máx. palavras na lista", 50, 2000, 300, 50)
    min_interval = st.slider("Intervalo entre chamadas (s)", 1.0, 6.0, 1.5, 0.5)

    if st.button("Limpar cache da matriz (sessão)"):
        st.session_state.pop("_matriz_cache", None)
        st.success("Cache da matriz limpo.")


arquivo = st.file_uploader("Envie a foto do Boggle (PNG/JPG)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 1])

with col1:
    if arquivo:
        img = Image.open(arquivo)
        st.image(img, caption="Imagem enviada", use_container_width=True)
    else:
        st.info("Envie uma imagem para começar.")

with col2:
    st.subheader("Resultado")

    if arquivo and st.button("🚀 Resolver", type="primary"):
        debounce_click(5)

        dicionario, prefixos = carregar_dicionario_pt()

        try:
            with st.spinner("Lendo a grade com a IA..."):
                matriz = extrair_matriz_imagem(img, api_key, min_interval=min_interval)

            st.write("**Matriz extraída:**")
            st.code(json.dumps({"matriz": matriz}, ensure_ascii=False, indent=2))

            with st.spinner("Buscando palavras..."):
                achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)

            # filtra tamanho mínimo
            palavras = [p for p in achadas.keys() if len(p) >= int(tam_minimo)]
            palavras.sort(key=lambda x: (-len(x), x))

            st.write(f"**Total de palavras (>= {tam_minimo}): {len(palavras)}**")

            # lista limitada
            palavras_show = palavras[: int(max_listar)]
            df = pd.DataFrame({"palavra": palavras_show, "tamanho": [len(p) for p in palavras_show]})
            st.dataframe(df, use_container_width=True, hide_index=True)

            # download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV", data=csv, file_name="palavras_boggle.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
