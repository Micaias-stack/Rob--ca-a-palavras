import base64
import hashlib
import io
import json
import re
import time
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai
from groq import Groq


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Robô Solver de Boggle", page_icon="🧩", layout="wide")
st.title("🧩 Robô Solver de Boggle (Google Gemini + Python)")
st.caption("Upload da foto → Gemini Vision extrai a grade → Python encontra as palavras (PT-BR).")

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
    achadas = {}
    direcoes = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    ]

    def dfs(r, c, visitados, palavra, caminho):
        letra = str(matriz[r][c]).upper().strip()
        nova_palavra = palavra + letra
        if nova_palavra not in prefixos:
            return
        if nova_palavra in dicionario and len(nova_palavra) >= 2 and nova_palavra not in achadas:
            achadas[nova_palavra] = list(caminho)
        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if 0 <= nr < linhas and 0 <= nc < colunas and (nr, nc) not in visitados:
                visitados_novo = visitados | {(nr, nc)}
                dfs(nr, nc, visitados_novo, nova_palavra, caminho + [(nr, nc)])

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])
    return achadas

# =========================================================
# UTILS: JSON & MATRIX
# =========================================================
def extrair_json_estrito(texto: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", (texto or "").strip())
    if not match:
        raise ValueError(f"Formato de resposta inválido. JSON não encontrado.\nRetorno: {texto[:500]}")
    return json.loads(match.group(0))

def normalizar_matriz(matriz, n):
    if not isinstance(matriz, list) or len(matriz) != n:
        raise ValueError(f"A matriz retornada pela IA é inválida (esperado {n} linhas).")
    out = []
    for i, row in enumerate(matriz):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"A linha {i} da matriz é inválida (deveria ter {n} colunas).")
        out.append([str(x).upper().strip() for x in row])
    return out

# =========================================================
# GOOGLE GEMINI: Extrair matriz da imagem
# =========================================================
def extrair_matriz_google(api_key: str, imagem: Image.Image, n: int):
    if not api_key:
        raise ValueError("A GOOGLE_API_KEY não foi configurada nos Secrets do Streamlit.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = (
        f"Analise a imagem de um tabuleiro do jogo Boggle. "
        f"Extraia as letras e retorne uma matriz {n}x{n} em formato JSON. "
        "As letras devem ser lidas da esquerda para a direita, de cima para baixo. "
        "Algumas células podem conter mais de uma letra (ex: 'QU', 'CH'). Mantenha-as juntas. "
        "Não inclua NADA além do objeto JSON na sua resposta. "
        f"Exemplo para um tabuleiro 4x4: {{\"matriz\":[[\"A\",\"B\",\"C\",\"D\"],[\"E\",\"F\",\"G\",\"H\"],[\"I\",\"J\",\"K\",\"L\"],[\"M\",\"N\",\"O\",\"P\"]]}}"
    )
    
    img_resized = imagem.copy()
    img_resized.thumbnail((800, 800))

    response = model.generate_content([prompt, img_resized], request_options={"timeout": 120})
    return extrair_json_estrito(response.text)

# =========================================================
# MAIN APP
# =========================================================
try:
    dicionario_pt, prefixos_pt = carregar_dicionario_pt()
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") # Mantido caso precise no futuro
except Exception as e:
    st.error(f"Erro na inicialização: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Envie a Foto do Tabuleiro")
    uploaded_file = st.file_uploader(
        "Arraste ou selecione a imagem (.jpg, .png)", type=["jpg", "png", "jpeg"]
    )
    tamanho_n = st.selectbox("Tamanho do tabuleiro:", [4, 5], index=0)
    
    if uploaded_file:
        imagem = Image.open(uploaded_file)
        st.image(imagem, caption="Imagem carregada.", use_column_width=True)

with col2:
    st.subheader("2. Resultado")
    if uploaded_file:
        if st.button(f"Analisar Tabuleiro {tamanho_n}x{tamanho_n}", use_container_width=True):
            placeholder_resultados = st.empty()
            with placeholder_resultados.container():
                try:
                    with st.spinner("🔍 Analisando a imagem com Google Gemini..."):
                        start_time = time.time()
                        json_resposta = extrair_matriz_google(GOOGLE_API_KEY, imagem, tamanho_n)
                        matriz = normalizar_matriz(json_resposta.get("matriz", []), tamanho_n)
                        st.write("✅ Matriz extraída pela IA:")
                        st.dataframe(pd.DataFrame(matriz), use_container_width=True)

                    with st.spinner("🧠 Procurando palavras no dicionário..."):
                        palavras_achadas = buscar_palavras_boggle(matriz, dicionario_pt, prefixos_pt)
                        end_time = time.time()
                        
                        st.success(f"🎉 {len(palavras_achadas)} palavras encontradas em {end_time - start_time:.2f} segundos!")
                        
                        # Organiza e exibe as palavras por tamanho
                        palavras_ordenadas = sorted(palavras_achadas.keys(), key=lambda p: (-len(p), p))
                        
                        df_palavras = pd.DataFrame({
                            'Palavra': palavras_ordenadas,
                            'Tamanho': [len(p) for p in palavras_ordenadas]
                        })
                        st.dataframe(df_palavras, use_container_width=True, height=400)

                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")

#### 5. Faça o Deploy
Faça `commit` e `push` dos arquivos `app.py` e `requirements.txt` atualizados para o seu repositório no GitHub. O Streamlit Cloud vai automaticamente reconstruir o app com as novas dependências e código.

Agora seu app está usando uma solução de visão de ponta, é leve, e vai funcionar perfeitamente no Streamlit Cloud.
