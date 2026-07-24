# app.py
import json
import re
import time
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
# PONTUAÇÃO (Boggle clássico)
# =========================================================
def pontos_boggle(tam: int) -> int:
    if tam <= 2:
        return 0
    if tam in (3, 4):
        return 1
    if tam == 5:
        return 2
    if tam == 6:
        return 3
    if tam == 7:
        return 5
    return 11  # 8+


# =========================================================
# BOGGLE SOLVER (DFS)
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0

    achadas = {}  # palavra -> caminho
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

    # bloco ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())

    # primeiro objeto { ... }
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"JSON não encontrado na resposta do modelo.\nResposta (início): {t[:400]}")
    return json.loads(m.group(0))


def _cell_ok(x: str) -> bool:
    s = (x or "").strip().upper()
    return bool(re.search(r"[A-ZÀ-Ü]", s))


def sanear_matriz(matriz):
    if not isinstance(matriz, list):
        return matriz

    out = []
    for row in matriz:
        if not isinstance(row, list):
            continue
        row2 = [str(c).upper().strip() for c in row]
        if any(_cell_ok(c) for c in row2):
            out.append(row2)

    return out


def ajustar_dimensoes(matriz, linhas: int, colunas: int):
    if not isinstance(matriz, list):
        raise ValueError("Matriz inválida: não é uma lista.")

    if len(matriz) < linhas:
        raise ValueError(f"Matriz inválida: esperado {linhas} linhas, veio {len(matriz)} (faltando linhas).")

    if len(matriz) > linhas:
        st.warning(f"O modelo retornou {len(matriz)} linhas; vou considerar apenas as primeiras {linhas}.")
        matriz = matriz[:linhas]

    fixed = []
    for i, row in enumerate(matriz):
        if not isinstance(row, list):
            raise ValueError(f"Linha {i} inválida: não é uma lista.")
        if len(row) < colunas:
            raise ValueError(f"Linha {i} inválida: esperado {colunas} colunas, veio {len(row)} (faltando colunas).")
        if len(row) > colunas:
            st.warning(f"A linha {i} veio com {len(row)} colunas; vou considerar apenas as primeiras {colunas}.")
            row = row[:colunas]
        fixed.append([str(x).upper().strip() for x in row])

    return fixed


# =========================================================
# GEMINI: Seleção robusta de modelo
# =========================================================
def _escolher_modelo_generate_content():
    preferidos = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    modelos = list(genai.list_models())
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


def _prompt_extracao(linhas, colunas):
    return (
        f"Você receberá a imagem de um tabuleiro tipo Boggle com {linhas} linhas e {colunas} colunas.\n"
        "Extraia as letras de cada célula.\n"
        "Regras:\n"
        "- Leia da esquerda para a direita, de cima para baixo.\n"
        "- Se uma célula tiver múltiplas letras (ex: 'QU'), mantenha junto.\n"
        "- Retorne SOMENTE JSON válido, sem texto extra.\n"
        f"- Formato: {{\"matriz\": [[...], ...]}} com exatamente {linhas} linhas e {colunas} colunas.\n"
    )


def _prompt_correcao(linhas, colunas, matriz_ruim):
    return (
        "Corrija a matriz abaixo para ficar EXATAMENTE no tamanho pedido.\n"
        f"Tamanho obrigatório: {linhas} linhas x {colunas} colunas.\n"
        "Regras:\n"
        "- Não invente letras novas.\n"
        "- Remova linhas vazias/repetidas e remova colunas extras.\n"
        "- Se houver uma linha de 'título' ou lixo, remova.\n"
        "- Retorne SOMENTE JSON válido no formato {\"matriz\": [[...], ...]}.\n\n"
        f"MATRIZ_ATUAL:\n{json.dumps({'matriz': matriz_ruim}, ensure_ascii=False)}\n"
    )


def extrair_matriz_google(api_key: str, imagem: Image.Image, linhas: int, colunas: int):
    if not api_key:
        raise ValueError("A GOOGLE_API_KEY não foi configurada nos Secrets do Streamlit.")

    genai.configure(api_key=api_key)

    modelo_id = _escolher_modelo_generate_content()
    model = genai.GenerativeModel(modelo_id)

    img_resized = imagem.copy()
    img_resized.thumbnail((1200, 1200))

    # 1) primeira tentativa
    prompt = _prompt_extracao(linhas, colunas)
    response = model.generate_content([prompt, img_resized], request_options={"timeout": 120})
    j = extrair_json_estrito(response.text)

    matriz = sanear_matriz(j.get("matriz"))

    # 2) se ainda estiver errado, pede correção (retry)
    try:
        matriz_ok = ajustar_dimensoes(matriz, linhas, colunas)
        return modelo_id, {"matriz": matriz_ok}
    except Exception:
        # retry usando a matriz atual como base
        prompt2 = _prompt_correcao(linhas, colunas, matriz)
        response2 = model.generate_content(prompt2, request_options={"timeout": 120})
        j2 = extrair_json_estrito(response2.text)
        matriz2 = sanear_matriz(j2.get("matriz"))
        matriz_ok2 = ajustar_dimensoes(matriz2, linhas, colunas)
        return modelo_id, {"matriz": matriz_ok2}


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
    uploaded_file = st.file_uploader("Selecione a imagem (.jpg, .png)", type=["jpg", "png", "jpeg"])

    # ✅ Inclui 5x6 e 6x5
    opcoes_tamanho = ["4x4", "5x5", "5x4", "6x4", "5x6", "6x5"]
    tamanho_selecionado = st.selectbox("Tamanho do tabuleiro (Linhas x Colunas):", opcoes_tamanho, index=0)

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
                    matriz = json_resposta["matriz"]
                    t1 = time.time()

                st.success(f"Matriz {linhas}x{colunas} extraída em {t1 - t0:.1f}s.")
                st.info(f"Modelo Gemini em uso: {modelo_usado}")
                st.code(json.dumps(matriz, indent=2, ensure_ascii=False), language="json")

                with st.spinner("🧠 Buscando palavras no dicionário PT-BR..."):
                    t2 = time.time()
                    palavras_achadas = buscar_palavras_boggle(matriz, dicionario_pt, prefixos_pt)
                    t3 = time.time()

                # ✅ Só 4 letras pra frente
                palavras_filtradas = {p: caminho for p, caminho in palavras_achadas.items() if len(p) >= 4}

                if not palavras_filtradas:
                    st.warning("Nenhuma palavra de 4+ letras encontrada com o dicionário atual.")
                    st.stop()

                # ordena por pontos, depois tamanho, depois alfabético
                palavras = sorted(
                    palavras_filtradas.keys(),
                    key=lambda w: (pontos_boggle(len(w)), len(w), w),
                    reverse=True
                )

                df = pd.DataFrame({
                    "palavra": palavras,
                    "tamanho": [len(w) for w in palavras],
                    "pontos": [pontos_boggle(len(w)) for w in palavras],
                })

                st.success(f"Encontrei {len(palavras)} palavras (4+) em {t3 - t2:.1f}s.")
                st.metric("Pontos totais", int(df["pontos"].sum()))
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Erro: {e}")
                st.text("Detalhes:")
                st.text(traceback.format_exc())
