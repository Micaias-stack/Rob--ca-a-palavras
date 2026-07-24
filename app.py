import json
import re
import time
import traceback
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# (Opcional) “cotidiano mas não comum” via wordfreq
try:
    from wordfreq import zipf_frequency
    WORDFREQ_OK = True
except Exception:
    WORDFREQ_OK = False


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Robô Caça-Palavras (Boggle)", page_icon="🧩", layout="wide")
st.title("🧩 Robô Caça-Palavras (Gemini + Python)")
st.caption("Upload da foto → Gemini extrai a grade → Python encontra palavras (PT-BR).")


# =========================================================
# REGRAS / MELHORIAS
# =========================================================
TAMANHOS = {"4x4": 4, "5x5": 5, "6x6": 6}  # apenas 4/5/6
MIN_PALAVRA = 3                            # ✅ mínimo 3 letras

# Palavra-chave: regras por tamanho do tabuleiro
def _min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def escolher_palavra_chave(palavras_encontradas: list[str], n_tabuleiro: int) -> str:
    """
    Palavra-chave:
    - prioriza palavras longas (por tabuleiro)
    - se wordfreq existir: tenta “cotidiano mas não muito comum”
      (faixa Zipf ~ 3.0 a 4.6, alvo 3.8)
    """
    if not palavras_encontradas:
        return ""

    min_len = _min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras_encontradas if len(w) >= min_len]

    if not cand:
        maxlen = max(len(w) for w in palavras_encontradas)
        cand = [w for w in palavras_encontradas if len(w) == maxlen]

    if WORDFREQ_OK:
        def z(w: str) -> float:
            return float(zipf_frequency(w.lower(), "pt"))

        # “cotidiano mas não comum”
        cand_mid = [w for w in cand if 3.0 <= z(w) <= 4.6]
        if cand_mid:
            cand = cand_mid

        alvo = 3.8
        def score(w: str):
            return (len(w), -abs(z(w) - alvo))

        return max(cand, key=score)

    # sem wordfreq: pega a maior (estável)
    cand.sort(key=lambda w: (-len(w), w))
    return cand[0]


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
        p = (p or "").upper().strip()
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

        # ✅ melhoria: mínimo 3 letras
        if nova in dicionario and len(nova) >= MIN_PALAVRA and nova not in achadas:
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
    """
    Remove linhas totalmente vazias e corta espaços.
    Não força dimensões ainda.
    """
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
    """
    Ajuste conservador:
    - se tiver linhas/colunas a mais: corta (com aviso)
    - se tiver a menos: erro (não inventa letra)
    """
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

def _get_api_key():
    # mantém compatível com seu deploy
    return st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")

def extrair_matriz_google(imagem: Image.Image, linhas: int, colunas: int):
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("A GOOGLE_API_KEY (ou GEMINI_API_KEY) não foi configurada nos Secrets do Streamlit.")

    genai.configure(api_key=api_key)

    modelo_id = _escolher_modelo_generate_content()
    model = genai.GenerativeModel(modelo_id)

    img_resized = imagem.copy()
    img_resized.thumbnail((1400, 1400))

    # 1) primeira tentativa
    resp1 = model.generate_content(
        [_prompt_extracao(linhas, colunas), img_resized],
        request_options={"timeout": 120},
    )
    j1 = extrair_json_estrito(resp1.text)
    m1 = sanear_matriz(j1.get("matriz"))

    # tenta ajustar já
    try:
        m_ok = ajustar_dimensoes(m1, linhas, colunas)
        return modelo_id, m_ok, j1
    except Exception:
        pass

    # 2) correção usando a matriz retornada
    resp2 = model.generate_content(
        [_prompt_correcao(linhas, colunas, m1), img_resized],
        request_options={"timeout": 120},
    )
    j2 = extrair_json_estrito(resp2.text)
    m2 = sanear_matriz(j2.get("matriz"))
    m_ok2 = ajustar_dimensoes(m2, linhas, colunas)
    return modelo_id, m_ok2, j2


# =========================================================
# UI: entrada
# =========================================================
with st.sidebar:
    st.header("Configurações")
    tamanho = st.selectbox("Tamanho do tabuleiro", list(TAMANHOS.keys()), index=0)
    st.caption(f"Mínimo de letras por palavra: **{MIN_PALAVRA}** (fixo)")

    st.divider()
    max_exibir = st.slider("Máximo de palavras para listar", 50, 2000, 300, 50)

st.divider()

arquivo = st.file_uploader("Envie a foto do tabuleiro", type=["png", "jpg", "jpeg", "webp"])

if not arquivo:
    st.stop()

imagem = Image.open(arquivo).convert("RGB")
st.image(imagem, caption="Imagem enviada", use_container_width=True)

n = TAMANHOS[tamanho]

# =========================================================
# Pipeline: extrair → resolver → palavra-chave → listar
# =========================================================
try:
    t0 = time.time()
    with st.spinner("Extraindo a grade com o Gemini..."):
        modelo_id, matriz, raw_json = extrair_matriz_google(imagem, n, n)
    t1 = time.time()

    st.success(f"Grade extraída com sucesso. Modelo: {modelo_id} | Tempo: {t1 - t0:.2f}s")

    st.subheader("📋 Grade reconhecida")
    df_grade = pd.DataFrame(matriz)
    st.dataframe(df_grade, use_container_width=True)

    dicionario, prefixos = carregar_dicionario_pt()

    with st.spinner("Buscando palavras (DFS)..."):
        t2 = time.time()
        achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)
        t3 = time.time()

    palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))
    st.info(f"Encontradas: {len(palavras)} palavras | Tempo solver: {t3 - t2:.2f}s")

    # ✅ melhoria: palavra-chave automática
    palavra_chave = escolher_palavra_chave(palavras, n)

    st.subheader("🎯 Palavra-chave (automática)")
    if palavra_chave:
        extra = ""
        if WORDFREQ_OK:
            z = float(zipf_frequency(palavra_chave.lower(), "pt"))
            extra = f" (Zipf≈{z:.2f})"
        st.write(f"**{palavra_chave}**{extra}")
    else:
        st.write("Não foi possível sugerir uma palavra-chave nesta rodada.")

    st.subheader("🔎 Palavras encontradas")
    if palavras:
        dados = []
        for w in palavras[:max_exibir]:
            dados.append({
                "palavra": w,
                "tamanho": len(w),
                "é_palavra_chave": (w == palavra_chave),
            })
        st.dataframe(pd.DataFrame(dados), use_container_width=True)
    else:
        st.write("Nenhuma palavra encontrada (com mínimo de 3 letras).")

    with st.expander("Debug (JSON bruto retornado pelo modelo)"):
        st.json(raw_json)

except Exception as e:
    st.error(str(e))
    with st.expander("Stacktrace"):
        st.code(traceback.format_exc())
