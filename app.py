# app.py
import io
import json
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import streamlit as st
from PIL import Image

import google.generativeai as genai

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Robô Caça-Palavras (Boggle)", layout="wide")
st.title("Robô Caça-Palavras (Boggle) — Gemini + Solver rápido")
st.caption("Foto → Gemini extrai a grade → TRIE filtrada + bitmask encontra palavras (PT-BR).")

TAMANHOS = {"4x4": 4, "5x5": 5, "6x6": 6}
MIN_PALAVRA = 3


# =========================
# NORMALIZAÇÃO
# =========================
def fold_upper(s: str) -> str:
    s = (s or "").strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


# =========================
# GEMINI: PROMPTS + PARSING
# =========================
def _prompt_extracao(linhas: int, colunas: int) -> str:
    return f"""
Você é um extrator de grade de letras (caça-palavras/Boggle).
Extraia APENAS a matriz {linhas}x{colunas} de letras do tabuleiro na imagem.

Regras:
- Responda SOMENTE em JSON válido (sem markdown, sem texto extra).
- Formato: {{"matriz":[["A","B",...], ...]}}
- Cada célula é 1 ou 2 letras (ex: "QU"), sempre string.
- Não invente letras que não estiverem visíveis. Se não der para ler, use "".
"""

def _prompt_correcao(linhas: int, colunas: int, matriz_parcial: list[list[str]]) -> str:
    return f"""
Você recebeu uma matriz parcial/irregular. Ajuste para ter EXATAMENTE {linhas} linhas e {colunas} colunas.
Regras:
- Responda SOMENTE em JSON válido.
- Formato: {{"matriz":[...]}}

Matriz atual:
{json.dumps({"matriz": matriz_parcial}, ensure_ascii=False)}
"""

def extrair_json_estrito(texto: str) -> dict[str, Any]:
    if not texto:
        raise ValueError("Resposta vazia do modelo.")

    # tenta direto
    try:
        return json.loads(texto)
    except Exception:
        pass

    # tenta capturar o primeiro objeto JSON
    m = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if not m:
        raise ValueError("Não encontrei JSON na resposta do modelo.")
    return json.loads(m.group(0))


def sanear_matriz(matriz: Any) -> list[list[str]]:
    if not isinstance(matriz, list):
        raise ValueError("Campo 'matriz' não é uma lista.")

    out: list[list[str]] = []
    for row in matriz:
        if not isinstance(row, list):
            raise ValueError("Linha da matriz não é lista.")
        out_row: list[str] = []
        for cell in row:
            s = fold_upper(str(cell)) if cell is not None else ""
            # deixa no máximo 2 letras (padrão comum do QU)
            if len(s) > 2:
                s = s[:2]
            out_row.append(s)
        out.append(out_row)
    return out


def ajustar_dimensoes(m: list[list[str]], linhas: int, colunas: int) -> list[list[str]]:
    # ajusta número de linhas
    m2 = m[:linhas] + ([[]] * max(0, linhas - len(m)))

    # ajusta colunas por linha
    out = []
    for r in range(linhas):
        row = m2[r] if isinstance(m2[r], list) else []
        row2 = (row[:colunas] + ([""] * max(0, colunas - len(row))))
        out.append(row2)
    return out


# =========================
# GEMINI: QUOTA (429) + FALLBACK MODELOS
# =========================
def _is_429(e: Exception) -> bool:
    s = str(e).lower()
    return ("429" in s) or ("quota" in s) or ("exceeded" in s) or ("resource exhausted" in s)

def _parse_retry_seconds(msg: str, default: int = 16) -> int:
    m = re.search(r"retry in\s*([\d\.]+)", msg, flags=re.IGNORECASE)
    if m:
        try:
            return max(1, int(float(m.group(1))))
        except Exception:
            pass
    m = re.search(r"retry_delay.*?seconds[:\s]*([0-9]+)", msg, flags=re.IGNORECASE)
    if m:
        try:
            return max(1, int(m.group(1)))
        except Exception:
            pass
    return default


def _img_to_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def extrair_matriz_google(imagem: Image.Image, linhas: int, colunas: int):
    api_key = (
        st.secrets.get("GOOGLE_API_KEY", "")
        or st.secrets.get("GEMINI_API_KEY", "")
    )
    if not api_key:
        raise ValueError("Configure GOOGLE_API_KEY (ou GEMINI_API_KEY) em st.secrets.")

    genai.configure(api_key=api_key)

    modelos_preferidos = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    img_resized = imagem.copy()
    img_resized.thumbnail((1400, 1400))

    last_err = None

    for modelo_id in modelos_preferidos:
        try:
            model = genai.GenerativeModel(modelo_id)

            for tent in range(3):
                try:
                    resp1 = model.generate_content(
                        [_prompt_extracao(linhas, colunas), img_resized],
                        request_options={"timeout": 120},
                    )
                    j1 = extrair_json_estrito(resp1.text)
                    m1 = ajustar_dimensoes(sanear_matriz(j1.get("matriz")), linhas, colunas)

                    # segunda passada: correção (opcional) se vier muito ruim
                    resp2 = model.generate_content(
                        [_prompt_correcao(linhas, colunas, m1)],
                        request_options={"timeout": 120},
                    )
                    j2 = extrair_json_estrito(resp2.text)
                    m2 = ajustar_dimensoes(sanear_matriz(j2.get("matriz")), linhas, colunas)

                    return modelo_id, m2, j2

                except Exception as e:
                    last_err = e
                    if _is_429(e) and tent < 2:
                        wait_s = _parse_retry_seconds(str(e), default=16)
                        st.warning(f"Limite temporário (429). Tentando novamente em {wait_s}s…")
                        time.sleep(wait_s)
                        continue
                    raise

        except Exception as e:
            last_err = e
            if _is_429(e):
                st.warning(f"Cota/limite no modelo {modelo_id}. Tentando outro modelo…")
                continue

    raise RuntimeError(f"Falha ao extrair matriz (todos modelos). Último erro: {last_err}")


@st.cache_data(ttl=86400, show_spinner=False)  # 24h
def extrair_matriz_google_cached(image_bytes: bytes, linhas: int, colunas: int):
    imagem = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return extrair_matriz_google(imagem, linhas, colunas)


# =========================
# FALLBACK MANUAL (colar matriz)
# =========================
def parse_matriz_manual(texto: str):
    t = (texto or "").strip()
    if not t:
        return None

    # tenta JSON
    try:
        j = extrair_json_estrito(t)
        m = j.get("matriz")
        return sanear_matriz(m)
    except Exception:
        pass

    # tenta “linhas”
    linhas = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\s;]+", line) if p]
        linhas.append(parts)
    return sanear_matriz(linhas)


# =========================
# WORDLIST + TRIE FILTRADA (o que realmente acelera)
# =========================
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def carregar_wordlist_ptbr():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    with urllib.request.urlopen(url, timeout=60) as resp:
        linhas = resp.read().decode("utf-8").splitlines()

    out = []
    for p in linhas:
        p = (p or "").strip()
        if not p or "-" in p or "." in p:
            continue
        w = fold_upper(p)
        if len(w) >= MIN_PALAVRA and re.fullmatch(r"[A-Z]+", w):
            out.append(w)
    return out


@dataclass(slots=True)
class TrieNode:
    children: dict = field(default_factory=dict)
    is_word: bool = False
    word: str | None = None


def trie_insert(root: TrieNode, w: str):
    node = root
    for ch in w:
        node = node.children.setdefault(ch, TrieNode())
    node.is_word = True
    node.word = w


def build_neigh(R: int, C: int):
    N = R * C
    neigh = [[] for _ in range(N)]
    for r in range(R):
        for c in range(C):
            i = r * C + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        neigh[i].append(nr * C + nc)
    return neigh


def bigramas_do_tabuleiro(cells: list[str], neigh: list[list[int]]):
    # bigramas possíveis olhando adjacência real do tabuleiro
    bg = set()
    for i, s1 in enumerate(cells):
        if not s1:
            continue
        a = s1[-1]  # última letra da célula (ex: "QU" termina em U)
        for j in neigh[i]:
            s2 = cells[j]
            if not s2:
                continue
            b = s2[0]  # primeira letra da célula vizinha
            bg.add(a + b)
    return bg


def filtrar_palavras_por_tabuleiro(wordlist: list[str], cells: list[str], neigh: list[list[int]], min_len: int, max_len: int):
    letras = set("".join(cells))
    bg = bigramas_do_tabuleiro(cells, neigh)

    cand = []
    for w in wordlist:
        L = len(w)
        if L < min_len or L > max_len:
            continue
        if not set(w) <= letras:
            continue
        ok = True
        for k in range(L - 1):
            if (w[k] + w[k + 1]) not in bg:
                ok = False
                break
        if ok:
            cand.append(w)
    return cand


def construir_trie(wordlist_filtrada: list[str]):
    root = TrieNode()
    max_len = 0
    for w in wordlist_filtrada:
        trie_insert(root, w)
        if len(w) > max_len:
            max_len = len(w)
    return root, max_len


# =========================
# SOLVER (bitmask + poda)
# =========================
def buscar_palavras_boggle(matriz: list[list[str]], trie_root: TrieNode, max_len_dict: int, min_palavra: int = 3):
    R = len(matriz)
    C = len(matriz[0]) if R else 0
    N = R * C
    if N == 0:
        return {}

    cap_por_tabuleiro = {4: 12, 5: 14, 6: 14}  # 6x6 mais agressivo pra ficar rápido
    MAX_LEN = min(max_len_dict, cap_por_tabuleiro.get(R, max_len_dict))

    cells = [fold_upper(str(matriz[r][c])) for r in range(R) for c in range(C)]
    neigh = build_neigh(R, C)

    achadas = {}
    path = []

    def step(i: int, node: TrieNode, visited_mask: int, depth: int):
        node2 = node
        d = depth

        s = cells[i]
        for ch in s:
            node2 = node2.children.get(ch)
            if node2 is None:
                return
            d += 1
            if d > MAX_LEN:
                return

        path.append(i)

        if node2.is_word:
            w = node2.word
            if w and len(w) >= min_palavra and w not in achadas:
                achadas[w] = [(idx // C, idx % C) for idx in path]

        if d < MAX_LEN:
            children_keys = node2.children  # poda pelo próximo início
            for j in neigh[i]:
                if (visited_mask >> j) & 1:
                    continue
                s2 = cells[j]
                if not s2:
                    continue
                if s2[0] not in children_keys:
                    continue
                step(j, node2, visited_mask | (1 << j), d)

        path.pop()

    # inicia só em células que existem no trie
    for i in range(N):
        node = trie_root
        ok = True
        for ch in cells[i]:
            node = node.children.get(ch)
            if node is None:
                ok = False
                break
        if ok:
            step(i, trie_root, (1 << i), 0)

    return achadas


def escolher_palavra_chave(palavras: list[str], n_tabuleiro: int) -> str:
    if not palavras:
        return ""
    min_len = {4: 6, 5: 7, 6: 8}.get(n_tabuleiro, 7)
    cand = [w for w in palavras if len(w) >= min_len] or palavras
    cand.sort(key=lambda w: (-len(w), w))
    return cand[0]


# =========================
# UI
# =========================
with st.sidebar:
    st.subheader("Configurações")
    tamanho = st.selectbox("Tamanho do tabuleiro", list(TAMANHOS.keys()), index=0)
    n = TAMANHOS[tamanho]
    usar_gemini = st.checkbox("Extrair com Gemini (imagem)", value=True)

    st.divider()
    st.caption("Chave no Streamlit Secrets:")
    st.code('GOOGLE_API_KEY="SUA_CHAVE_AQUI"', language="toml")


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Entrada")
    arquivo = st.file_uploader("Envie a foto do tabuleiro (PNG/JPG)", type=["png", "jpg", "jpeg"])
    texto_manual = st.text_area(
        "Fallback: cole a matriz (JSON ou linhas). Ex:\nA B C D\nE F G H\nI J K L\nM N O P",
        height=140,
    )

with col2:
    st.subheader("Saída")
    matriz = None

    if arquivo:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="Imagem enviada", use_container_width=True)

        if usar_gemini:
            try:
                with st.spinner("Extraindo matriz com Gemini…"):
                    image_bytes = _img_to_bytes(imagem)
                    modelo_id, matriz_ext, _raw = extrair_matriz_google_cached(image_bytes, n, n)
                    matriz = ajustar_dimensoes(matriz_ext, n, n)
                st.success(f"Matriz extraída ({modelo_id}).")
            except Exception as e:
                if _is_429(e):
                    st.error("Cota/limite da API (429). Use o campo de fallback manual abaixo para colar a matriz.")
                else:
                    st.error(f"Falha ao extrair a matriz: {e}")

    # fallback manual sempre disponível (prioriza se Gemini falhou ou se não enviou arquivo)
    if matriz is None:
        m_manual = parse_matriz_manual(texto_manual)
        if m_manual:
            matriz = ajustar_dimensoes(m_manual, n, n)
            st.info("Usando matriz do fallback manual.")

    if matriz:
        st.write("Matriz final:")
        st.dataframe(matriz, use_container_width=True, hide_index=True)

        with st.spinner("Carregando dicionário…"):
            wordlist = carregar_wordlist_ptbr()

        # prepara filtro por tabuleiro (isso acelera de verdade)
        R = len(matriz)
        C = len(matriz[0]) if R else 0
        cells = [fold_upper(str(matriz[r][c])) for r in range(R) for c in range(C)]
        neigh = build_neigh(R, C)

        cap_por_tabuleiro = {4: 12, 5: 14, 6: 14}
        max_len = cap_por_tabuleiro.get(R, 14)

        t0 = time.perf_counter()
        filtradas = filtrar_palavras_por_tabuleiro(wordlist, cells, neigh, MIN_PALAVRA, max_len)
        trie_root, max_len_dict = construir_trie(filtradas)
        t1 = time.perf_counter()

        with st.spinner("Procurando palavras…"):
            achadas = buscar_palavras_boggle(matriz, trie_root, max_len_dict, min_palavra=MIN_PALAVRA)
        t2 = time.perf_counter()

        palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))
        chave = escolher_palavra_chave(palavras, R)

        st.write(f"- Palavras candidatas após filtro: **{len(filtradas)}**")
        st.write(f"- Tempo filtro+TRIE: **{(t1 - t0):.2f}s**")
        st.write(f"- Tempo busca: **{(t2 - t1):.2f}s**")
        st.write(f"- Total encontradas: **{len(palavras)}**")

        if chave:
            st.markdown(f"**Palavra-chave sugerida:** `{chave}`")

        st.divider()
        st.subheader("Top palavras")
        st.write(palavras[:50] if palavras else "Nenhuma palavra encontrada com as regras atuais.")

    else:
        st.info("Envie a imagem (e/ou cole a matriz manualmente) para rodar o solver.")
