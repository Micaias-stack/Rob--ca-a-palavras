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
from time import time
import hashlib

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

    tam_minimo = st.number_input("Tamanho mínimo das palavras", min_value=2, max_value=20, value=3, step=1)
    max_listar = st.slider("Máx. palavras para listar", min_value=50, max_value=1000, value=200, step=50)

# ==========================================
# UTILITÁRIOS DE COTA
# ==========================================
def qpm_guard(min_interval=1.5):
    agora = time()
    ultimo = st.session_state.get("_last_req_ts", 0)
    espera = min_interval - (agora - ultimo)
    if espera > 0:
        time.sleep(espera)
    st.session_state["_last_req_ts"] = time()

def debounce_click(intervalo=5):
    agora = time()
    ultimo = st.session_state.get("_last_click_ts", 0)
    if agora - ultimo < intervalo:
        faltam = intervalo - (agora - ultimo)
        st.warning(f"Aguarde {faltam:.1f}s antes de tentar de novo.")
        st.stop()
    st.session_state["_last_click_ts"] = agora

# ==========================================
# MOTOR DICIONÁRIO PT-BR (CACHE)
# ==========================================
@st.cache_data(ttl=21600)  # 6h
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
# IMAGEM → BASE64 + HASH (para cache local por foto)
# ==========================================
def b64_e_hash_da_imagem(imagem):
    img_temp = imagem.copy().convert("RGB")
    img_temp.thumbnail((800, 800))
    buf = io.BytesIO()
    img_temp.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("utf-8"), hashlib.sha256(raw).hexdigest()

# ==========================================
# LISTAGEM DE MODELOS (CACHE 6h)
# ==========================================
@st.cache_data(ttl=21600)
def listar_modelos(api_key: str):
    models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    qpm_guard()  # evita estourar QPM na própria listagem
    res = requests.get(models_url, timeout=60)
    res.raise_for_status()
    data = res.json()
    return [
        m["name"]
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]

# ==========================================
# EXTRAÇÃO VIA REST API (usa modelos com sufixo "-image")
# ==========================================
def extrair_matriz_imagem(imagem, api_key):
    import re

    # Cache por imagem
    if "_matriz_cache" not in st.session_state:
        st.session_state["_matriz_cache"] = {}

    img_b64, img_hash = b64_e_hash_da_imagem(imagem)
    if img_hash in st.session_state["_matriz_cache"]:
        return st.session_state["_matriz_cache"][img_hash]

    # Listar modelos (com cache)
    try:
        modelos_gc = listar_modelos(api_key)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Erro ao contatar a API do Google para listar modelos: {e}")

    # Somente modelos de visão: sufixo '-image' (inclui '-image-preview', etc.)
    modelos_image = [n for n in modelos_gc if re.search(r"(^|-)image($|[-])", n)]

    # Preferências (prioriza os mais novos/estáveis quando disponíveis)
    preferencias = [
        "models/gemini-3.1-flash-image",
        "models/gemini-3.1-flash-lite-image",
        "models/gemini-3-pro-image",
        "models/gemini-3.1-flash-image-preview",
        "models/gemini-3-pro-image-preview",
        "models/gemini-2.5-flash-image",
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
        ],
        "generationConfig": {"maxOutputTokens": 128, "temperature": 0.2},
    }

    # Retentativa com Retry-After (429/503) + guarda de QPM
    max_retries = 5
    response = None
    for attempt in range(max_retries):
        qpm_guard()  # espaça chamadas para não estourar RPM
        try:
            response = requests.post(generate_url, json=payload, timeout=120)
            if response.status_code == 200:
                break

            if response.status_code in (429, 503) and attempt < max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds = int(retry_after)
                else:
                    wait_seconds = (2 ** attempt) + random.uniform(0, 1.5)
                st.info(
                    f"Limite atingido ({response.status_code}). Retentando em {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
                continue

            # Sai do loop se não for 200/429/503 ou acabou as tentativas
            break
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Erro de conexão com a API: {str(e)}")

    if response is None:
        raise ValueError("Falha na conexão com a API após várias tentativas.")
    if response.status_code != 200:
        raise ValueError(f"Erro na requisição ({response.status_code}): {response.text}")

    # Parse do JSON retornado (alguns modelos devolvem texto com o JSON embutido)
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

    # Cache por imagem (não gasta cota se enviar a mesma foto)
    st.session_state["_matriz_cache"][img_hash] = dados["matriz"]
    return dados["matriz"]

# ==========================================
# APLICAÇÃO
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
            if st.button("🚀 Destruir no Boggle", type="primary", use_container_width=True):
                debounce_click(5)  # evita duplo clique/estouro de RPM
                with st.spinner("Lendo a grade com a IA..."):
                    try:
                        matriz = extrair_matriz_imagem(imagem, api_key)
                        st.success("Grade identificada!")

                        # Mostrar a grade detectada
                        df_grid = pd.DataFrame(matriz)
                        st.write("Grade reconhecida:")
                        st.dataframe(df_grid, use_container_width=True, height=min(400, 50 * len(matriz) + 70))

                        # Buscar palavras
                        st.info("Calculando todas as palavras possíveis…")
                        palavras = buscar_palavras_boggle(matriz, dicionario, prefixos)

                        # Filtrar e ordenar
                        lista = [p for p in palavras.keys() if len(p) >= tam_minimo]
                        lista.sort(key=lambda x: (-len(x), x))

                        total = len(lista)
                        maior = max(lista, key=len) if lista else "-"
                        st.success(f"Encontradas {total} palavras (mín. {tam_minimo} letras). Maior: {maior}")

                        # Listagem limitada para não travar no celular
                        st.write(f"Mostrando até {max_listar} palavras:")
                        st.write(", ".join(lista[:max_listar]))

                    except Exception as e:
                        st.error(f"Ocorreu um erro: {str(e)}")
