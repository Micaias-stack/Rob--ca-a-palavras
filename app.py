import json
import re
import time
import traceback
import unicodedata
import urllib.request
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# opcional: “raridade” (se estiver instalado, ajuda a escolher a palavra-chave)
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
# REGRAS DO JOGO (UPGRADES)
# =========================================================
MIN_PALAVRA = 3                 # ✅ mínimo para encontrar palavras
TAMANHOS_PERMITIDOS = {"4x4", "5x5", "6x6"}  # ✅ só 4x4, 5x5, 6x6

PONTOS_MAX_UNICA = 8            # cap da pontuação por tamanho (únicas)
PONTOS_MAX_CHAVE = 8            # palavra-chave vale até 8 (fixo, editável no slider)
PONTOS_REPETIDA = 1             # palavra repetida (2+ jogadores)


# =========================================================
# NORMALIZAÇÃO (mais robusto pra acento/ruído do OCR)
# =========================================================
def strip_accents(s: str) -> str:
    s = s or ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def canon_word(s: str) -> str:
    """
    Canônico:
    - remove acentos
    - upper
    - fica só A-Z
    """
    s = strip_accents(s or "").upper().strip()
    s = re.sub(r"[^A-Z]", "", s)
    return s

def zipf_pt_safe(word: str) -> float:
    if not WORDFREQ_OK:
        return 0.0
    return float(zipf_frequency((word or "").lower(), "pt"))


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
        p = (p or "").strip()
        if not p or "-" in p or "." in p:
            continue

        c = canon_word(p)
        if len(c) >= 2:
            dicionario.add(c)
            for i in range(1, len(c) + 1):
                prefixos.add(c[:i])

    return dicionario, prefixos


# =========================================================
# BOGGLE SOLVER (DFS)
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos, min_palavra=MIN_PALAVRA):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0

    achadas = {}  # palavra -> caminho
    direcoes = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def dfs(r, c, visitados, palavra, caminho):
        letra = canon_word(str(matriz[r][c]))
        if not letra:
            return

        nova = palavra + letra

        if nova not in prefixos:
            return

        if nova in dicionario and len(nova) >= min_palavra and nova not in achadas:
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
    s = canon_word(x)
    return len(s) >= 1

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

def extrair_matriz_google(api_key: str, imagem: Image.Image, linhas: int, colunas: int):
    if not api_key:
        raise ValueError("A GOOGLE_API_KEY não foi configurada nos Secrets do Streamlit.")

    genai.configure(api_key=api_key)

    modelo_id = _escolher_modelo_generate_content()
    model = genai.GenerativeModel(modelo_id)

    img_resized = imagem.copy()
    img_resized.thumbnail((1400, 1400))

    # 1) primeira tentativa
    resp1 = model.generate_content([_prompt_extracao(linhas, colunas), img_resized], request_options={"timeout": 120})
    j1 = extrair_json_estrito(resp1.text)
    m1 = sanear_matriz(j1.get("matriz"))

    try:
        m_ok = ajustar_dimensoes(m1, linhas, colunas)
        return modelo_id, m_ok, j1
    except Exception:
        pass

    # 2) retry: correção de dimensões
    resp2 = model.generate_content([_prompt_correcao(linhas, colunas, m1), img_resized], request_options={"timeout": 120})
    j2 = extrair_json_estrito(resp2.text)
    m2 = sanear_matriz(j2.get("matriz"))

    m_ok = ajustar_dimensoes(m2, linhas, colunas)
    return modelo_id, m_ok, j2


# =========================================================
# UPGRADE: PALAVRA-CHAVE AUTOMÁTICA (4x4/5x5/6x6)
# =========================================================
def min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def sugerir_palavra_chave(palavras_canon: list[str], n_tabuleiro: int) -> str:
    """
    Escolhe automaticamente:
    - prioritiza as mais longas a partir do mínimo por tabuleiro
    - desempate por raridade (se wordfreq existir)
    """
    if not palavras_canon:
        return ""

    min_len = min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras_canon if len(w) >= min_len]

    if not cand:
        maxlen = max(len(w) for w in palavras_canon)
        cand = [w for w in palavras_canon if len(w) == maxlen]

    def score(w: str) -> float:
        raridade = -zipf_pt_safe(w)  # menor frequência => mais rara => score maior
        return (len(w) * 100.0) + (raridade * 10.0)

    return max(cand, key=score)


# =========================================================
# UPGRADE: PONTUAÇÃO (opcional, pra você colar palavras dos jogadores)
# =========================================================
def pontos_unica_por_tamanho(n: int) -> int:
    if n <= 2: return 0
    if n == 3: return 2
    if n == 4: return 3
    if n == 5: return 4
    if n == 6: return 6
    return PONTOS_MAX_UNICA  # 7+ => 8

def pontuar_palavra(canon: str, qtd_jogadores_que_acharam: int, palavra_chave_canon: str, pontos_chave: int) -> int:
    if palavra_chave_canon and canon == palavra_chave_canon:
        return int(pontos_ch
