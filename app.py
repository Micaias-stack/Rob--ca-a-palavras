# app.py — FOTO -> Gemini (Streamlit secrets) -> grade 4x4/5x5/6x6 -> Solver + Pontuação
#
# ✅ A API key vem do Streamlit: st.secrets["GEMINI_API_KEY"] (sem campo de input)
# ✅ Reconhece automaticamente a grade (4x4, 5x5 ou 6x6)
# ✅ Palavra-chave automática por tabuleiro (mais longa + mais rara, se wordfreq existir)
# ✅ Pontuação:
#   - Repetida (2+ jogadores): 1 ponto
#   - Única: por tamanho (cap em 8)
#   - Palavra-chave: fixa (0..8), default 8
#
# Requisitos:
#   pip install streamlit pandas pillow google-generativeai
# Opcional:
#   pip install wordfreq
#
# Rodar:
#   streamlit run app.py

import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# opcional: melhora a escolha “rara”
try:
    from wordfreq import zipf_frequency
    WORDFREQ_OK = True
except Exception:
    WORDFREQ_OK = False


# =========================================================
# CONFIG
# =========================================================
PONTOS_MAX_UNICA = 8
PONTOS_MAX_CHAVE = 8
TAMANHOS_PERMITIDOS = {4, 5, 6}


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Caça-Palavras — Foto + Solver + Pontuação", layout="wide")
st.title("Caça-Palavras — Foto + Solver + Pontuação")

with st.sidebar:
    st.header("Configurações")

    min_len = st.slider("Mínimo de letras (válidas)", 3, 10, 3, 1)
    max_exibir = st.slider("Máximo de palavras para listar", 50, 2000, 300, 50)

    st.divider()
    pontos_chave = st.slider("Pontos da palavra-chave", 0, PONTOS_MAX_CHAVE, PONTOS_MAX_CHAVE, 1)
    n_jog = st.number_input("Quantidade de jogadores", min_value=2, max_value=10, value=2, step=1)


# =========================================================
# Normalização
# =========================================================
def strip_accents(s: str) -> str:
    s = s or ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def canon_word(s: str) -> str:
    s = strip_accents(s or "").upper().strip()
    s = re.sub(r"[^A-Z]", "", s)
    return s

def zipf_pt_safe(word: str) -> float:
    if not WORDFREQ_OK:
        return 0.0
    return float(zipf_frequency((word or "").lower(), "pt"))


# =========================================================
# Dicionário PT-BR (canon + prefixos + formas)
# =========================================================
@st.cache_data(ttl=21600)
def carregar_dicionario_pt_canon():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    with urllib.request.urlopen(url, timeout=60) as resp:
        palavras_raw = resp.read().decode("utf-8", errors="ignore").splitlines()

    dicionario = set()
    prefixos = set()
    canon_to_forms = defaultdict(set)

    for p in palavras_raw:
        p0 = (p or "").strip()
        if not p0:
            continue
        if any(ch.isdigit() for ch in p0) or "." in p0 or "-" in p0 or "'" in p0:
            continue

        c = canon_word(p0)
        if len(c) < 2:
            continue

        dicionario.add(c)
        canon_to_forms[c].add(p0)

        for i in range(1, len(c) + 1):
            prefixos.add(c[:i])

    return dicionario, prefixos, dict(canon_to_forms)

def escolher_melhor_forma(canon: str, canon_to_forms: dict) -> str:
    forms = list(canon_to_forms.get(canon, []))
    if not forms:
        return canon

    if WORDFREQ_OK:
        forms.sort(key=lambda w: zipf_pt_safe(w), reverse=True)
        return forms[0]

    return sorted(forms)[0]


# =========================================================
# Solver DFS (8 direções) — aceita célula com 1+ letras (ex.: "QU")
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    n = len(matriz)
    achadas = {}  # canon -> caminho
    direcoes = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def dfs(r, c, visitados, palavra, caminho):
        cel = canon_word(str(matriz[r][c]))
        if not cel:
            return

        nova = palavra + cel
        if nova not in prefixos:
            return

        if nova in dicionario and nova not in achadas:
            achadas[nova] = list(caminho)

        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visitados:
                dfs(nr, nc, visitados | {(nr, nc)}, nova, caminho + [(nr, nc)])

    for r in range(n):
        for c in range(n):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return achadas


# =========================================================
# Pontuação
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
        return int(pontos_chave)
    if qtd_jogadores_que_acharam >= 2:
        return 1
    return pontos_unica_por_tamanho(len(canon))


# =========================================================
# Palavra-chave automática (longa + rara)
# =========================================================
def min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def sugerir_palavra_chave(palavras_canon: list[str], canon_to_forms: dict, n_tabuleiro: int) -> str:
    if not palavras_canon:
        return ""

    min_len = min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras_canon if len(w) >= min_len]

    if not cand:
        maxlen = max((len(w) for w in palavras_canon), default=0)
        cand = [w for w in palavras_canon if len(w) == maxlen]

    def score(canon: str) -> float:
        form = escolher_melhor_forma(canon, canon_to_forms)
        raridade = -zipf_pt_safe(form)
        return (len(canon) * 100.0) + (raridade * 10.0)

    melhor = max(cand, key=score)
    return escolher_melhor_forma(melhor, canon_to_forms)


# =========================================================
# Gemini: extrair grade 4/5/6 da imagem
# =========================================================
def extrair_json_estrito(texto: str) -> dict:
    t = (texto or "").strip()
    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("JSON não encontrado na resposta do Gemini.")
    return json.loads(m.group(0))

def limpar_cell(cell: str) -> str:
    cell = (cell or "").strip().upper()
    cell = re.sub(r"[^A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]", "", cell)
    return cell

def validar_grade(grid):
    if not isinstance(grid, list) or not grid:
        raise ValueError("Grade inválida (não é lista).")

    n = len(grid)
    if n not in TAMANHOS_PERMITIDOS:
        raise ValueError(f"Tamanho detectado {n}x{n} não permitido. Só 4x4, 5x5, 6x6.")

    for row in grid:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("Grade inválida (linhas com tamanho diferente).")

    out = []
    for r in range(n):
        row2 = []
        for c in range(n):
            row2.append(limpar_cell(str(grid[r][c])))
        out.append(row2)

    return out

def extrair_grade_por_gemini(img: Image.Image):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError('Faltou configurar st.secrets["GEMINI_API_KEY"].')

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """
Extraia APENAS a grade de letras do caça-palavras da imagem.
A grade é sempre quadrada e pode ser 4x4, 5x5 ou 6x6.

Retorne SOMENTE JSON (sem texto fora do JSON) no formato:
{"grid":[["A","B","C","D"], ...]}

Regras:
- Use 1 célula por posição.
- Se a célula tiver "QU", retorne "QU" (sem espaço).
- Não invente linhas/colunas.
"""

    resp = model.generate_content([prompt, img])
    data = extrair_json_estrito(resp.text)
    grid = data.get("grid")
    return validar_grade(grid)


# =========================================================
# APP
# =========================================================
dicionario, prefixos, canon_to_forms = carregar_dicionario_pt_canon()

img_file = st.file_uploader("Envie a foto do tabuleiro (4x4, 5x5 ou 6x6)", type=["png", "jpg", "jpeg", "webp"])

if not img_file:
    st.stop()

img = Image.open(img_file).convert("RGB")
st.image(img, caption="Imagem enviada", use_container_width=True)

with st.spinner("Reconhecendo a grade (Gemini) e resolvendo..."):
    grade = extrair_grade_por_gemini(img)
    n = len(grade)

    achadas = buscar_palavras_boggle(grade, dicionario, prefixos)
    # aplica mínimo de letras
    achadas = {w: path for w, path in achadas.items() if len(w) >= int(min_len)}

    palavra_chave = sugerir_palavra_chave(list(achadas.keys()), canon_to_forms, n)
    palavra_chave_canon = canon_word(palavra_chave)

# --- exibição da grade
st.subheader(f"Grade detectada: {n}x{n}")
st.dataframe(pd.DataFrame(grade), use_container_width=True, hide_index=True)

# --- resumo + palavra-chave
col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Palavras encontradas", f"{len(achadas)}")
col2.metric("Mín. letras", f"{min_len}")
col3.write(f"**Palavra-chave (automática):** {palavra_chave}  \n**Valor:** {pontos_chave} pontos (fixo)")

# --- lista das palavras (limitada)
st.subheader("Lista (limitada)")
formas = [escolher_melhor_forma(w, canon_to_forms) for w in achadas.keys()]
formas_ordenadas = sorted(formas, key=lambda x: (-len(canon_word(x)), x.lower()))
st.write(", ".join(formas_ordenadas[: int(max_exibir)]) or "—")

st.divider()

# =========================================================
# Placar multi-jogadores
# =========================================================
st.subheader("Placar (jogadores)")

entradas = {}
for i in range(int(n_jog)):
    nome = st.text_input(f"Nome do jogador {i+1}", value=f"Jogador {i+1}", key=f"nome_{i}")
    texto = st.text_area(f"Palavras do {nome} (1 por linha)", height=120, key=f"pal_{i}")
    entradas[nome] = texto

if st.button("Calcular placar"):
    # nome -> lista (canon)
    canon_por_jogador = {}
    for nome, texto in entradas.items():
        lista = [ln.strip() for ln in (texto or "").splitlines() if ln.strip()]
        canon_por_jogador[nome] = [canon_word(p) for p in lista if len(canon_word(p)) >= 3]

    # conta em quantos jogadores apareceu (não duplica dentro do mesmo jogador)
    contagem = Counter()
    for lst in canon_por_jogador.values():
        contagem.update(set(lst))

    # calcula
    placar = {}
    detalhes = {}
    for nome, lst in canon_por_jogador.items():
        total = 0
        det = []
        for w in sorted(set(lst), key=lambda x: (-len(x), x)):
            if w not in achadas:
                # se quiser aceitar palavra fora do solver, apague esse bloco
                continue
            pts = pontuar_palavra(w, contagem[w], palavra_chave_canon, pontos_chave)
            total += pts
            det.append((escolher_melhor_forma(w, canon_to_forms), pts, contagem[w]))
        placar[nome] = total
        detalhes[nome] = det

    st.write("**Placar:**")
    st.dataframe(
        pd.DataFrame(
            [{"Jogador": n, "Pontos": p} for n, p in sorted(placar.items(), key=lambda x: x[1], reverse=True)]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Detalhes (palavra / pontos / nº de jogadores que acharam):**")
    for nome in detalhes:
        st.markdown(f"**{nome}**")
        if not detalhes[nome]:
            st.write("—")
            continue
        st.dataframe(
            pd.DataFrame(detalhes[nome], columns=["Palavra", "Pontos", "Qtd jogadores"]),
            use_container_width=True,
            hide_index=True,
        )
