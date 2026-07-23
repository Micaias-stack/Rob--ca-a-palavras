import base64
import io
import json
import re
import urllib.request
import pandas as pd
import requests
from PIL import Image
import streamlit as st
import time
import random

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(page_title="Robô Hacker de Boggle", page_icon="🧩", layout="wide")
st.title("🧩 Robô Solver de Boggle (Netflix)")
st.markdown(
    "Carregue a foto da TV. O robô vai ler a grade (inclusive blocos como CH) e calcular **TODAS** as palavras possíveis (das maiores até as de 2 e 3 letras)!"
)

with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

# ==========================================
# MOTOR DICIONÁRIO PT-BR (CACHE)
# ==========================================
@st.cache_data
def carregar_dicionario_pt():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    try:
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
    except Exception as e:
        st.error(f"Erro ao baixar dicionário: {str(e)}")
        return set(), set()

# ==========================================
# MOTOR DE BUSCA ZIGUE-ZAGUE (DFS)
# ==========================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas > 0 else 0
    palavras_encontradas = {}

    def dfs(r, c, visitados, palavra_atual, caminho):
        letra_celula = str(matriz[r][c]).upper().strip()
        nova_palavra = palavra_atual + letra_celula

        if nova_palavra not in prefixos:
            return

        if nova_palavra in dicionario and len(nova_palavra) >= 2:
            if nova_palavra not in palavras_encontradas:
                palavras_encontradas[nova_palavra] = list(caminho)

        direcoes = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if 0 <= nr < linhas and 0 <= nc < colunas and (nr, nc) not in visitados:
                visitados.add((nr, nc))
                dfs(nr, nc, visitados, nova_palavra, caminho + [(nr, nc)])
                visitados.remove((nr, nc))

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return palavras_encontradas

# ==========================================
# REDIMENSIONAR E CONVERTER IMAGEM
# ==========================================
def imagem_para_base64_otimizada(imagem):
    img_temp = imagem.copy().convert("RGB")
    img_temp.thumbnail((800, 800))
    buffered = io.BytesIO()
    img_temp.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# EXTRAÇÃO VIA REST API (SELEÇÃO DE MODELO *-image + RETRY 503)
# ==========================================
def extrair_matriz_imagem(imagem, api_key):
    # 1) Listar modelos disponíveis na conta
    models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        res_models = requests.get(models_url, timeout=60)
        res_models.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Erro ao contatar a API do Google para listar modelos: {e}")

    models_data = res_models.json()
    modelos_gc = [
        m["name"]
        for m in models_data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]

    # 2) Manter apenas modelos com sufixo '-image' (inclui '-image-preview', '-image-...'):
    modelos_image = [n for n in modelos_gc if re.search(r"(^|-)image($|[-])", n)]

    # 3) Preferências priorizando os mais novos/estáveis
    preferencias = [
        "models/gemini-3.1-flash-image",
        "models/gemini-3-pro-image",
        "models/gemini-2.5-flash-image",
        "models/gemini-3.1-flash-lite-image",
        "models/gemini-3.1-flash-image-preview",
        "models/gemini-3-pro-image-preview",
    ]
    modelo_escolhido = next((p for p in preferencias if p in modelos_image), None)
    if not modelo_escolhido and modelos_image:
        modelo_escolhido = modelos_image[0]

    if not modelo_escolhido:
        raise ValueError(
            "❌ Nenhum modelo de visão encontrado nesta chave (sufixo '-image'). "
            f"Modelos com generateContent: {modelos_gc}"
        )

    st.info(f"Usando o modelo de IA: `{modelo_escolhido}`")

    # 4) Montar payload
    img_b64 = imagem_para_base64_otimizada(imagem)
    generate_url = f"https://generativelanguage.googleapis.com/v1beta/{modelo_escolhido}:generateContent?key={api_key}"

    prompt = (
        'Analise esta imagem de um jogo de palavras (Boggle). '
        'Extraia APENAS a matriz/grade completa de letras. '
        'MUITO IMPORTANTE: observe se algumas células possuem DUAS letras juntas '
        '(ex.: "CH", "QU", "RR"). Extraia exatamente como está no bloco '
        '(mantenha o "CH" na mesma string da célula).\n\n'
        'Responda EXCLUSIVAMENTE em JSON estrito, sem markdown:\n'
        '{"matriz":[["E","J","S","Z","Z","S"],["C","CH","I","F","O","S"]]}'
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                ]
            }
        ]
    }

    # 5) Retentativa com backoff exponencial (503)
    max_retries = 5
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(generate_url, json=payload, timeout=120)
            if response.status_code == 200:
                break
            if response.status_code == 503 and attempt < max_retries - 1:
                wait_seconds = (2 ** attempt) + random.uniform(0, 1)
                st.info(f"API sobrecarregada (503). Tentando novamente em {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)
                continue
            break
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Erro de conexão com a API: {str(e)}")

    if response is None:
        raise ValueError("Falha na conexão com a API após várias tentativas.")
    if response.status_code == 429:
        raise ValueError("⏳ Cota por minuto excedida. Aguarde 60s e tente novamente!")
    if response.status_code != 200:
        raise ValueError(f"Erro na requisição ({response.status_code}): {response.text}")

    # 6) Parse do JSON retornado
    data = response.json()
    try:
        texto_resposta = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise ValueError(f"Resposta inesperada da API: {json.dumps(data)}")

    match_json = re.search(r"\{.*\}", texto_resposta, re.DOTALL)
    if not match_json:
        raise ValueError("Falha ao ler o JSON da IA. Resposta recebida: " + texto_resposta)

    dados = json.loads(match_json.group())
    if "matriz" not in dados or not isinstance(dados["matriz"], list):
        raise ValueError("O JSON não contém a chave 'matriz' no formato esperado.")

    return dados["matriz"]

# ==========================================
# FLUXO PRINCIPAL
# ==========================================
dicionario, prefixos = carregar_dicionario_pt()

uploaded_file = st.file_uploader("Envie a foto da TV", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file and dicionario:
    imagem = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Imagem")
        st.image(imagem, use_column_width=True)

    with col2:
        st.subheader("⚙️ Status")
        if not api_key:
            st.warning("Insira sua Gemini API Key na barra lateral.")
        else:
            if st.button("🚀 Destruir no Boggle", type="primary"):
                with st.spinner("Lendo a grade com a IA..."):
                    try:
                        matriz = extrair_matriz_imagem(imagem, api_key)
                        st.success("Grade identificada!")

                        df_grid = pd.DataFrame(matriz)
                        st.subheader("📐 Grade reconhecida")
                        st.dataframe(df_grid, use_container_width=True)

                        # Buscar palavras
                        st.subheader("📚 Palavras encontradas")
                        resultados = buscar_palavras_boggle(matriz, dicionario, prefixos)
                        palavras_ordenadas = sorted(resultados.keys(), key=lambda w: (-len(w), w))
                        st.write(f"Total: {len(palavras_ordenadas)}")
                        max_show = 300
                        df_palavras = pd.DataFrame({"palavra": palavras_ordenadas[:max_show]})
                        st.dataframe(df_palavras, use_container_width=True, height=500)

                    except Exception as e:
                        st.error(f"Ocorreu um erro: {e}")
