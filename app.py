# app.py
import json
import re
import time
import traceback
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Robô Caça-Palavras (Boggle)", page_icon="🧩", layout="wide")
st.title("🧩 Robô Caça-Palavras (Gemini + Python)")
st.caption("Upload da foto → Gemini extrai a grade → Python encontra palavras (PT-BR).")


# =========================================================
# DICIONÁRIO PT-BR (CACHE)
# =========================================================
@st.cache_data(ttl=21600)  # 6h
def carregar_dicionario_pt():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    with urllib.request.urlopen(url, timeout=60) as resposta:
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


# =========================================================
# BOGGLE SOLVER (DFS)
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0

    achadas = {}  # palavra -> caminho
    direcoes = [  # 8 direções
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
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
                dfs(nr, nc, visitados | {(nr, nc)}, nova, caminho + [(nr, nc)])

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return achadas


# =========================================================
# UTILS: JSON & MATRIX
# =========================================================
def extrair_json_estrito(texto: str) -> dict:
    t = (texto or "").strip()

    # tenta pegar bloco ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        candidato = m.group(1).strip()
        return json.loads(candidato)

    # fallback: primeiro objeto { ... } encontrado
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"JSON não encontrado na resposta do modelo.\nResposta (início): {t[:400]}")
    return json.loads(m.group(0))


def normalizar_matriz(matriz, linhas: int, colunas: int):
    if not isinstance(matriz, list) or len(matriz) != linhas:
        raise ValueError(f"Matriz inválida: esperado {linhas} linhas, veio {len(matriz) if isinstance(matriz, list) else 'N/A'}.")

    out = []
    for i, row in enumerate(matriz):
        if not isinstance(row, list) or len(row) != colunas:
            raise ValueError(f"Linha {i} inválida: esperado {colunas} colunas, veio {len(row) if isinstance(row, list) else 'N/A'}.")
        out.append([str(x).upper().strip() for x in row])

    return out


# =========================================================
# GEMINI: Seleção robusta de modelo (sem “chute”)
# =========================================================
def _escolher_modelo_generate_content():
    preferidos = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    modelos = list(genai.list_models())
    # ex: m.name == "models/gemini-..."
    mapa = {m.name.replace("models/", ""): m for m in modelos}

    def suporta(m):
        methods = getattr(m, "supported_generation_methods", None) or []
        return "generateContent" in methods

    for mid in preferidos:
        m = mapa.get(mid)
        if m and suporta(m):
            return mid

    for m in modelos:
        if suporta(m):
            return m.name.replace("models/", "")

    raise RuntimeError("Nenhum modelo com suporte a generateContent foi encontrado no seu projeto (v1beta).")


def extrair_matriz_google(api_key: str, imagem: Image.Image, linhas: int, colunas: int):
    if not api_key:
        raise ValueError("A GOOGLE_API_KEY não foi configurada nos Secrets do Streamlit.")

    genai.configure(api_key=api_key)

    modelo_id = _escolher_modelo_generate_content()
    model = genai.GenerativeModel(modelo_id)

    prompt = (
        f"Você receberá a imagem de um tabuleiro tipo Boggle com {linhas} linhas e {colunas} colunas.\n"
        "Extraia as letras de cada célula.\n"
        "Regras:\n"
        "- Leia da esquerda para a direita, de cima para baixo.\n"
        "- Se uma célula tiver 'QU' (ou múltiplas letras), mantenha junto como uma string.\n"
        "- Retorne SOMENTE JSON válido, sem texto extra.\n"
        f"- Formato: {{\"matriz\": [[...], ...]}} com exatamente {linhas} linhas e {colunas} colunas.\n"
    )

    img_resized = imagem.copy()
    img_resized.thumbnail((1200, 1200))

    response = model.generate_content([prompt, img_resized], request_options={"timeout": 120})
    return modelo_id, extrair_json_estrito(response.text)


# =========================================================
# MAIN APP
# =========================================================
try:
    dicionario_pt, prefixos_pt = carregar_dicionario_pt()
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
except Exception as e:
    st.error(f"Erro na inicialização: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1) Envie a foto")
    uploaded_file = st.file_uploader(
        "Selecione a imagem (.jpg, .png)", type=["jpg", "png", "jpeg"]
    )

    opcoes_tamanho = ["4x4", "5x5", "5x4", "6x4"]
    tamanho_selecionado = st.selectbox(
        "Tamanho do tabuleiro (Linhas x Colunas):",
        opcoes_tamanho,
        index=0
    )

    if uploaded_file:
        imagem = Image.open(uploaded_file)
        st.image(imagem, caption="Imagem carregada", use_container_width=True)

with col2:
    st.subheader("2) Resultado")

    if uploaded_file:
        if st.button(f"Analisar tabuleiro {tamanho_selecionado}", use_container_width=True):
            try:
                linhas, colunas = map(int, tamanho_selecionado.split("x"))

                with st.spinner("🔍 Extraindo a matriz com Gemini..."):
                    t0 = time.time()
                    modelo_usado, json_resposta = extrair_matriz_google(GOOGLE_API_KEY, imagem, linhas, colunas)
                    matriz = normalizar_matriz(json_resposta.get("matriz"), linhas, colunas)
                    t1 = time.time()

                st.success(f"Matriz {linhas}x{colunas} extraída em {t1 - t0:.1f}s.")
                st.info(f"Modelo Gemini em uso: {modelo_usado}")
                st.code(json.dumps(matriz, indent=2, ensure_ascii=False), language="json")

                with st.spinner("🧠 Buscando palavras no dicionário PT-BR..."):
                    t2 = time.time()
                    palavras_achadas = buscar_palavras_boggle(matriz, dicionario_pt, prefixos_pt)
                    t3 = time.time()

                if not palavras_achadas:
                    st.warning("Nenhuma palavra encontrada com o dicionário atual.")
                else:
                    palavras = sorted(palavras_achadas.keys(), key=len, reverse=True)
                    st.success(f"Encontrei {len(palavras)} palavras em {t3 - t2:.1f}s.")

                    df = pd.DataFrame({
                        "Palavra": palavras,
                        "Tamanho": [len(p) for p in palavras],
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")
                st.code(traceback.format_exc(), language="text")
