# app.py — Caça-Palavras/Boggle (PT-BR) com:
# - Tabuleiro 4x4, 5x5, 6x6
# - Palavras válidas a partir de 3 letras
# - Solver (DFS 8 direções) com dicionário PT-BR
# - Pontuação multi-jogadores:
#   * Palavra repetida (2+ jogadores): 1 ponto
#   * Palavra única: 2..8 pontos por tamanho (cap em 8)
#   * Palavra-chave: muda por tabuleiro (sugerida automaticamente) e vale até 8 pontos (fixo)
#
# Requisitos:
#   pip install streamlit pandas pillow google-generativeai
# Opcional (melhora “raridade” da palavra-chave):
#   pip install wordfreq

import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# opcional: priorização por “raridade”
try:
    from wordfreq import zipf_frequency
    WORDFREQ_OK = True
except Exception:
    WORDFREQ_OK = False


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Caça-Palavras — Solver + Pontuação", layout="wide")
st.title("Caça-Palavras — Solver + Pontuação")

with st.sidebar:
    st.header("Configurações")

    n_tabuleiro = st.selectbox("Tamanho do tabuleiro", [4, 5, 6], index=2)
    min_len = st.slider("Mínimo de letras (válidas)", 3, 10, 3, 1)
    max_exibir = st.slider("Máximo de palavras para listar", 50, 2000, 300, 50)

    st.divider()
    st.subheader("Extração da grade por foto (Gemini)")
    api_key = st.text_input("GEMINI_API_KEY", type="password", value=st.secrets.get("GEMINI_API_KEY", ""))
    usar_gemini = st.toggle("Usar Gemini para extrair a grade", value=True)


# =========================================================
# Normalização
# =========================================================
def strip_accents(s: str) -> str:
    s = s or ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def canon_word(s: str) -> str:
    # CANÔNICO: sem acento, maiúsculo, só letras A-Z
    s = strip_accents(s or "").upper().strip()
    s = re.sub(r"[^A-Z]", "", s)
    return s

def zipf_pt_safe(word: str) -> float:
    if not WORDFREQ_OK:
        return 0.0
    return float(zipf_frequency((word or "").lower(), "pt"))


# =========================================================
# Dicionário PT-BR (canon + prefixos + mapeamento para formas)
# =========================================================
@st.cache_data(ttl=21600)
def carregar_dicionario_pt_canon():
    # Fonte simples e leve (pode trocar se quiser)
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

        # evita tokens estranhos
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
# Solver (DFS 8 direções)
# - aceita células com 1+ letras (ex.: "QU")
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0

    achadas = {}  # canon -> caminho (lista de (r,c))
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
            if 0 <= nr < linhas and 0 <= nc < colunas and (nr, nc) not in visitados:
                dfs(nr, nc, visitados | {(nr, nc)}, nova, caminho + [(nr, nc)])

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return achadas


# =========================================================
# Pontuação (max 8)
# =========================================================
PONTOS_MAX_UNICA = 8
PONTOS_MAX_CHAVE = 8

def pontos_unica_por_tamanho(n: int) -> int:
    # só usado quando a palavra foi "única" (apenas 1 jogador achou)
    if n <= 2: return 0
    if n == 3: return 2
    if n == 4: return 3
    if n == 5: return 4
    if n == 6: return 6
    # 7+ fica no teto
    return PONTOS_MAX_UNICA  # 8

def pontuar_palavra(canon: str, qtd_jogadores_que_acharam: int, palavra_chave_canon: str, pontos_chave: int) -> int:
    # Palavra-chave: pontuação fixa (0..8)
    if palavra_chave_canon and canon == palavra_chave_canon:
        return int(pontos_chave)

    # Repetida (2+ jogadores): 1 ponto
    if qtd_jogadores_que_acharam >= 2:
        return 1

    # Única: por tamanho (2..8)
    return pontos_unica_por_tamanho(len(canon))


# =========================================================
# Palavra-chave automática por tabuleiro
# - “mais letras” + (se disponível) “mais rara”
# =========================================================
def min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def sugerir_palavra_chave(palavras_canon: list[str], canon_to_forms: dict, n_tabuleiro: int) -> str:
    if not palavras_canon:
        return ""

    min_len = min_len_chave_por_tabuleiro(n_tabuleiro)

    # 1) candidatas longas
    cand = [w for w in palavras_canon if len(w) >= min_len]

    # 2) fallback: as mais longas do tabuleiro
    if not cand:
        maxlen = max((len(w) for w in palavras_canon), default=0)
        cand = [w for w in palavras_canon if len(w) == maxlen]

    def score(canon: str) -> float:
        form = escolher_melhor_forma(canon, canon_to_forms)
        raridade = -zipf_pt_safe(form)  # menor frequência => mais “rara”
        return (len(canon) * 100.0) + (raridade * 10.0)

    melhor = max(cand, key=score)
    return escolher_melhor_forma(melhor, canon_to_forms)


# =========================================================
# Entrada da grade (manual + imagem/Gemini)
# =========================================================
def ajustar_dimensoes(matriz, n: int):
    if not isinstance(matriz, list) or len(matriz) != n or any(not isinstance(r, list) or len(r) != n for r in matriz):
        raise ValueError(f"Matriz precisa ser {n}x{n}.")

    # normaliza células para string (aceita 'QU', etc.)
    out = []
    for r in range(n):
        row = []
        for c in range(n):
            cell = str(matriz[r][c]).strip().upper()
            cell = re.sub(r"[^A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]", "", cell)  # mantém acentos se vierem
            row.append(cell)
        out.append(row)
    return out

def parse_grade_texto(txt: str, n: int):
    linhas = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
    if len(linhas) != n:
        raise ValueError(f"Esperava {n} linhas; recebi {len(linhas)}.")

    matriz = []
    for ln in linhas:
        # aceita separado por espaço ou colado
        parts = ln.split()
        if len(parts) == n:
            row = parts
        else:
            # sem espaços: tenta quebrar caractere a caractere
            raw = ln.replace(" ", "")
            # aqui não dá pra adivinhar 'QU' colado; então exija espaços se usar dígrafos
            if len(raw) != n:
                raise ValueError("Linha inválida. Use espaços entre células (principalmente se tiver 'QU').")
            row = list(raw)
        matriz.append(row)

    return ajustar_dimensoes(matriz, n)

def extrair_json_estrito(texto: str) -> dict:
    t = (texto or "").strip()
    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("JSON não encontrado na resposta do modelo.")
    return json.loads(m.group(0))

def extrair_grade_gemini(img: Image.Image, n: int, api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
Extraia a grade de letras de um tabuleiro de caça-palavras estilo Boggle.
Regras:
- O tabuleiro é exatamente {n}x{n}.
- Cada célula deve ser uma string (ex: "A", "B", "QU").
- Responda SOMENTE com JSON no formato:
{{
  "grid": [["A","B",...], ...]
}}
Sem texto extra.
"""

    resp = model.generate_content([prompt, img])
    data = extrair_json_estrito(resp.text)
    if "grid" not in data:
        raise ValueError("JSON veio sem a chave 'grid'.")
    return ajustar_dimensoes(data["grid"], n)


# =========================================================
# Layout principal
# =========================================================
dicionario, prefixos, canon_to_forms = carregar_dicionario_pt_canon()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) Informe o tabuleiro")

    modo = st.radio("Modo de entrada", ["Manual", "Imagem"], horizontal=True)

    matriz = None

    if modo == "Manual":
        st.caption("Dica: use espaços entre as células. Ex (4x4):\nA B C D")
        exemplo = "\n".join([" ".join(["A"] * n_tabuleiro) for _ in range(n_tabuleiro)])
        txt = st.text_area("Cole a grade (uma linha por linha do tabuleiro)", value=exemplo, height=180)
        try:
            matriz = parse_grade_texto(txt, n_tabuleiro)
        except Exception as e:
            st.error(str(e))

    else:
        up = st.file_uploader("Envie uma imagem do tabuleiro", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            img = Image.open(up).convert("RGB")
            st.image(img, caption="Imagem recebida", use_container_width=True)

            if usar_gemini:
                if not api_key:
                    st.error("Preencha GEMINI_API_KEY na lateral.")
                else:
                    try:
                        matriz = extrair_grade_gemini(img, n_tabuleiro, api_key)
                    except Exception as e:
                        st.error(f"Falha ao extrair grade via Gemini: {e}")
            else:
                st.info("Ative o toggle do Gemini na lateral ou use o modo Manual.")

    if matriz:
        st.write("Grade interpretada:")
        st.dataframe(pd.DataFrame(matriz))

with col2:
    st.subheader("2) Solver (palavras possíveis)")
    if not matriz:
        st.info("Preencha a grade para rodar o solver.")
        st.stop()

    palavras_achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)
    # filtra por min_len
    palavras_achadas = {w: path for w, path in palavras_achadas.items() if len(w) >= min_len}

    st.write(f"Total encontradas (>= {min_len} letras): **{len(palavras_achadas)}**")

    # Lista para exibir (mais longas primeiro)
    itens = []
    for canon in palavras_achadas.keys():
        forma = escolher_melhor_forma(canon, canon_to_forms)
        itens.append((canon, forma, len(canon)))

    itens.sort(key=lambda x: (x[2], x[1].lower()), reverse=True)
    itens = itens[:max_exibir]

    df = pd.DataFrame(itens, columns=["CANON", "Palavra", "Letras"])
    st.dataframe(df[["Palavra", "Letras"]], use_container_width=True, height=360)

    st.divider()
    st.subheader("3) Palavra‑chave (automática por tabuleiro)")

    palavra_chave_auto = sugerir_palavra_chave(list(palavras_achadas.keys()), canon_to_forms, n_tabuleiro)
    palavra_chave = st.text_input("Palavra‑chave (pode editar)", value=palavra_chave_auto)
    pontos_chave = st.slider("Pontos da palavra‑chave", 0, PONTOS_MAX_CHAVE, PONTOS_MAX_CHAVE, 1)

    st.caption(
        f"Heurística: tamanho mínimo sugerido = {min_len_chave_por_tabuleiro(n_tabuleiro)} letras (com fallback para as mais longas)."
    )

st.divider()

# =========================================================
# Pontuação multi-jogadores
# =========================================================
st.subheader("4) Pontuação (jogadores)")

n_jog = st.number_input("Quantidade de jogadores", min_value=2, max_value=10, value=2, step=1)

entradas = {}
cols = st.columns(int(n_jog))
for i in range(int(n_jog)):
    with cols[i]:
        nome = st.text_input(f"Nome {i+1}", value=f"Jogador {i+1}", key=f"nome_{i}")
        texto = st.text_area("Palavras (1 por linha)", height=180, key=f"pal_{i}")
        entradas[nome] = texto

if st.button("Calcular placar"):
    # palavras válidas do tabuleiro
    validas_set = set(palavras_achadas.keys())

    # monta dict nome -> set de palavras canon válidas (>=3)
    canon_por_jogador = {}
    for nome, texto in entradas.items():
        lista = [ln.strip() for ln in (texto or "").splitlines() if ln.strip()]
        canon_list = []
        for p in lista:
            c = canon_word(p)
            if len(c) < 3:
                continue
            # só pontua se existir no tabuleiro
            if c in validas_set:
                canon_list.append(c)
        canon_por_jogador[nome] = set(canon_list)

    # conta em quantos jogadores cada palavra apareceu
    contagem = Counter()
    for s in canon_por_jogador.values():
        contagem.update(s)

    chave_canon = canon_word(palavra_chave) if palavra_chave else ""

    # calcula total + detalhamento
    rows = []
    for nome, s in canon_por_jogador.items():
        total = 0
        for w in s:
            total += pontuar_palavra(w, contagem[w], chave_canon, pontos_chave)
        rows.append({"Jogador": nome, "Pontos": total, "Qtd palavras válidas": len(s)})

    placar_df = pd.DataFrame(rows).sort_values(["Pontos", "Qtd palavras válidas"], ascending=[False, False])
    st.dataframe(placar_df, use_container_width=True)

    # detalhamento por jogador (curto)
    with st.expander("Ver detalhamento (pontos por palavra)"):
        for nome, s in canon_por_jogador.items():
            det = []
            for w in sorted(s, key=lambda x: (pontuar_palavra(x, contagem[x], chave_canon, pontos_chave), len(x), x), reverse=True):
                pts = pontuar_palavra(w, contagem[w], chave_canon, pontos_chave)
                det.append((escolher_melhor_forma(w, canon_to_forms), len(w), contagem[w], pts))
            ddf = pd.DataFrame(det, columns=["Palavra", "Letras", "Jogadores que acharam", "Pontos"])
            st.markdown(f"**{nome}**")
            st.dataframe(ddf, use_container_width=True, height=240)
