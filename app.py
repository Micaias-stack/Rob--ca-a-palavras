import io
import json
import re
import time
import urllib.request
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# (Opcional) "cotidiano mas não comum" via wordfreq
try:
    from wordfreq import zipf_frequency
    WORDFREQ_OK = True
except Exception:
    WORDFREQ_OK = False


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Robô Caça-Palavras (Boggle)", layout="wide")
st.title("Robô Caça-Palavras (Gemini + Python)")
st.caption("Upload da foto → Gemini extrai a grade → Python encontra palavras (PT-BR).")

TAMANHOS = {"4x4": 4, "5x5": 5, "6x6": 6}
MIN_PALAVRA = 3


# =========================================================
# PALAVRAS ACEITAS (REFERÊNCIA) — você enviou
# (a gente usa pra reforçar o dicionário e melhorar a seleção da palavra-chave)
# =========================================================
PALAVRAS_COMUNS_REF_TEXT = r"""
que, com, por, uma, nas, nos, das, dos, sem, aos, mas, pra, pro, tem, foi, sou, era, ser, ver, seu, sua, meu, teu, tua, ele
ela, nós, não, sim, nem, dez, mil, cem, vez, ano, dia, bem, mal, bom, boa, dor, cor, luz, sol, paz, fez, vai, vem, vou, usa
uso, sob, lhe, tal, num, uns, mim, ali, ora, oco, tão, mão, pão, são, dar, ler, rir, pôr, tia, tio, pai, mãe, avó, avô, faz
diz, lei, lua, mar, via, rua, voz, som, sai, viu, deu, cai, pôs, põe, ato, eis, cal, mel, sal, chá, céu, olá, eco, gás, sul

para, mais, como, onde, aqui, hoje, caso, pois, pela, pelo, este, esta, isto, isso, dele, dela, nele, nela, cada, algo, nada, tudo
vida, casa, lado, modo, hora, dias, anos, bens, ruas, ruim, alto, leve, doce, pode, quer, fala, fica, veja, você, eles, elas

muito, assim, ainda, pouco, tempo, gente, coisa, parte, vezes, nunca, sobre, fazer, poder, dever, deixa, dessa, desse, deste, mesmo
menos, agora, antes, noite, tarde, certo, seria, saber, achar, ficar, levar, pedir, dizer, falar, viver, mundo, livro, jovem, idoso

porque, depois, embora, apenas, sempre, jamais, dentro, abaixo, frente, apesar, pessoa, escola, amigos, amigas, comida, bebida, salada
cidade, estado, bairro, centro, planta, animal, imagem, tarefa, estudo, matriz, planos, tomada, seguro, mínimo, máximo, rápido, modelo

exemplo, pessoas, família, criança, preciso, precisa, caminho, momento, segundo, mercado, sistema, produto, serviço, cuidado, celular
projeto, reunião, viagens, domingo, segunda, limpeza, cozinha, cadeira, tijolos, amarelo, alegria, objetos, garrafa, sapatos, camisas

trabalho, problema, qualquer, controle, programa, telefone, internet, aprender, mensagem, resposta, pergunta, material, processo
negócios, empresas, mercados, sistemas, projetos, clientes, pessoais, famílias, crianças
"""


# =========================================================
# NORMALIZAÇÃO (remove acentos) — melhora muito o match no tabuleiro
# =========================================================
def fold_upper(s: str) -> str:
    s = (s or "").strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def parse_common_words(texto: str) -> set[str]:
    # pega tokens separados por vírgula/espacos/quebra e filtra por "palavra" (A-Z)
    raw = re.split(r"[^A-Za-zÀ-ÿ]+", texto or "")
    out = set()
    for t in raw:
        t2 = fold_upper(t)
        if len(t2) >= MIN_PALAVRA and re.fullmatch(r"[A-Z]+", t2 or ""):
            out.add(t2)
    return out

PALAVRAS_COMUNS_REF = parse_common_words(PALAVRAS_COMUNS_REF_TEXT)


# =========================================================
# UX: palavra-chave (cotidiana mas não óbvia)
# =========================================================
def _min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def escolher_palavra_chave(palavras_encontradas: list[str], n_tabuleiro: int) -> str:
    if not palavras_encontradas:
        return ""

    min_len = _min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras_encontradas if len(w) >= min_len]

    if not cand:
        maxlen = max(len(w) for w in palavras_encontradas)
        cand = [w for w in palavras_encontradas if len(w) == maxlen]

    # tenta evitar cair em palavra muito "óbvia" se tiver outra boa opção
    # (se sobrar só óbvia, tudo bem)
    non_obvious = [w for w in cand if w not in PALAVRAS_COMUNS_REF]
    if non_obvious:
        cand = non_obvious

    if WORDFREQ_OK:
        def z(w: str) -> float:
            # wordfreq costuma trabalhar melhor com minúsculas (mas sem acento também funciona razoavelmente)
            return float(zipf_frequency(w.lower(), "pt"))

        # “cotidiano mas não comum demais”
        cand_mid = [w for w in cand if 3.0 <= z(w) <= 4.6]
        if cand_mid:
            cand = cand_mid

        alvo = 3.8
        def score(w: str):
            return (len(w), -abs(z(w) - alvo))

        return max(cand, key=score)

    cand.sort(key=lambda w: (-len(w), w))
    return cand[0]


# =========================================================
# TRIE (rápido)
# =========================================================
@dataclass(slots=True)
class TrieNode:
    children: dict = field(default_factory=dict)
    is_word: bool = False
    word: str | None = None  # guarda palavra (folded)

def _trie_insert(root: TrieNode, w: str):
    node = root
    for ch in w:
        node = node.children.setdefault(ch, TrieNode())
    node.is_word = True
    node.word = w

@st.cache_resource
def carregar_trie_ptbr(min_len: int = 3, extras: frozenset[str] = frozenset()):
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    with urllib.request.urlopen(url, timeout=60) as resp:
        palavras_raw = resp.read().decode("utf-8").splitlines()

    root = TrieNode()
    max_len = 0

    # dicionário base
    for p in palavras_raw:
        p = (p or "").strip()
        if "-" in p or "." in p:
            continue

        p2 = fold_upper(p)
        if len(p2) < min_len:
            continue
        if not re.fullmatch(r"[A-Z]+", p2):
            continue

        _trie_insert(root, p2)
        max_len = max(max_len, len(p2))

    # reforço: palavras aceitas que você mandou
    for w in extras:
        w2 = fold_upper(w)
        if len(w2) >= min_len and re.fullmatch(r"[A-Z]+", w2):
            _trie_insert(root, w2)
            max_len = max(max_len, len(w2))

    return root, max_len


# =========================================================
# SOLVER ULTRA RÁPIDO (bitmask + vizinhos pré-computados)
# =========================================================
def buscar_palavras_boggle(matriz, trie_root: TrieNode, max_len_dict: int):
    R = len(matriz)
    C = len(matriz[0]) if R else 0
    N = R * C
    if N == 0:
        return {}

    # cap de profundidade (evita explodir no 6x6)
    cap_por_tabuleiro = {4: 12, 5: 14, 6: 16}
    MAX_LEN = min(max_len_dict, cap_por_tabuleiro.get(R, max_len_dict))

    # flatten + normaliza letras
    cells = []
    for r in range(R):
        for c in range(C):
            s = fold_upper(str(matriz[r][c]))
            cells.append(s)

    # vizinhos (8 direções)
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

    achadas = {}  # palavra -> caminho (lista de (r,c))
    path = []     # guarda índices

    def step(i: int, node: TrieNode, visited_mask: int, depth: int):
        node2 = node
        d = depth

        # aplica letras da célula (ex: "QU")
        for ch in cells[i]:
            node2 = node2.children.get(ch)
            if node2 is None:
                return
            d += 1
            if d > MAX_LEN:
                return

        path.append(i)

        if node2.is_word:
            w = node2.word
            if w and len(w) >= MIN_PALAVRA and w not in achadas:
                achadas[w] = [(idx // C, idx % C) for idx in path]

        if d < MAX_LEN:
            for j in neigh[i]:
                if (visited_mask >> j) & 1:
                    continue
                step(j, node2, visited_mask | (1 << j), d)

        path.pop()

    # inicia somente em células que casam com início de alguma palavra
    for i in range(N):
        node = trie_root
        ok = True
        for ch in cells[i]:
            node = node.children.get(ch)
            if node is None:
                ok = False
                break
        if not ok:
            continue
        step(i, trie_root, (1 << i), 0)

    return achadas


# =========================================================
# GEMINI: prompts + robustez de quota + cache por imagem
# =========================================================
def extrair_json_estrito(texto: str) -> dict:
    t = (texto or "").strip()

    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"JSON não encontrado na resposta do modelo. Início: {t[:300]}")
    return json.loads(m.group(0))

def sanear_matriz(matriz):
    if not isinstance(matriz, list):
        return matriz
    out = []
    for row in matriz:
        if not isinstance(row, list):
            continue
        row2 = [str(c).strip() for c in row]
        if any((c or "").strip() for c in row2):
            out.append(row2)
    return out

def ajustar_dimensoes(matriz, linhas: int, colunas: int):
    if not isinstance(matriz, list):
        raise ValueError("Matriz inválida: não é uma lista.")
    if len(matriz) < linhas:
        raise ValueError(f"Matriz inválida: esperado {linhas} linhas, veio {len(matriz)}.")
    if len(matriz) > linhas:
        st.warning(f"O modelo retornou {len(matriz)} linhas; vou considerar apenas as primeiras {linhas}.")
        matriz = matriz[:linhas]

    fixed = []
    for i, row in enumerate(matriz):
        if not isinstance(row, list):
            raise ValueError(f"Linha {i} inválida: não é uma lista.")
        if len(row) < colunas:
            raise ValueError(f"Linha {i} inválida: esperado {colunas} colunas, veio {len(row)}.")
        if len(row) > colunas:
            st.warning(f"A linha {i} veio com {len(row)} colunas; vou considerar apenas as primeiras {colunas}.")
            row = row[:colunas]
        fixed.append([str(x).strip() for x in row])
    return fixed

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
    # manda a matriz ruim de volta pro modelo ajustar
    return (
        f"Corrija a matriz abaixo para ficar EXATAMENTE {linhas}x{colunas}.\n"
        "Regras:\n"
        "- Não invente letras; apenas ajuste alinhamento, removendo vazios/ruídos.\n"
        "- Retorne SOMENTE JSON válido no formato: {\"matriz\": [[...]]}\n"
        f"Matriz atual: {json.dumps({'matriz': matriz_ruim}, ensure_ascii=False)}\n"
    )

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
    return default

def _img_to_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_data(ttl=86400, show_spinner=False)
def extrair_matriz_google_cached(image_bytes: bytes, linhas: int, colunas: int):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return extrair_matriz_google(img, linhas, colunas)

def extrair_matriz_google(imagem: Image.Image, linhas: int, colunas: int):
    api_key = st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Configure GOOGLE_API_KEY (ou GEMINI_API_KEY) nos Secrets do Streamlit.")

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

            # tentativa (com retry leve se 429)
            for tent in range(3):
                try:
                    resp1 = model.generate_content(
                        [_prompt_extracao(linhas, colunas), img_resized],
                        request_options={"timeout": 120},
                    )
                    j1 = extrair_json_estrito(resp1.text)
                    m1 = sanear_matriz(j1.get("matriz"))

                    try:
                        m_ok = ajustar_dimensoes(m1, linhas, colunas)
                        return modelo_id, m_ok, j1
                    except Exception:
                        pass

                    # correção
                    resp2 = model.generate_content(
                        [_prompt_correcao(linhas, colunas, m1)],
                        request_options={"timeout": 120},
                    )
                    j2 = extrair_json_estrito(resp2.text)
                    m2 = sanear_matriz(j2.get("matriz"))
                    m_ok2 = ajustar_dimensoes(m2, linhas, colunas)
                    return modelo_id, m_ok2, j2

                except Exception as e:
                    last_err = e
                    if _is_429(e) and tent < 2:
                        wait_s = _parse_retry_seconds(str(e), default=16)
                        time.sleep(wait_s)
                        continue
                    raise

        except Exception as e:
            last_err = e
            if _is_429(e):
                continue

    raise RuntimeError(f"Falha ao extrair a matriz (todos os modelos). Último erro: {last_err}")


# =========================================================
# FALLBACK MANUAL (quando estourar quota)
# =========================================================
def parse_matriz_manual(texto: str):
    t = (texto or "").strip()
    if not t:
        return None

    # JSON
    try:
        j = extrair_json_estrito(t)
        m = sanear_matriz(j.get("matriz"))
        return m
    except Exception:
        pass

    # Linhas
    linhas = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\s;]+", line) if p]
        linhas.append(parts)

    return sanear_matriz(linhas)


# =========================================================
# UI
# =========================================================
with st.sidebar:
    tamanho = st.selectbox("Tamanho do tabuleiro", list(TAMANHOS.keys()), index=1)
    n = TAMANHOS[tamanho]
    st.write(f"- Mínimo de letras: **{MIN_PALAVRA}**")
    st.write("- Solver rápido: **TRIE + bitmask**")
    usar_gemini = st.toggle("Usar Gemini pra extrair da imagem", value=True)

col1, col2 = st.columns([1, 1])

with col1:
    up = st.file_uploader("Envie a foto do tabuleiro", type=["png", "jpg", "jpeg", "webp"])
    if up:
        imagem = Image.open(up).convert("RGB")
        st.image(imagem, caption="Imagem enviada", use_container_width=True)

with col2:
    st.subheader("Matriz detectada / manual")
    matriz = None
    modelo_usado = None

    if up and usar_gemini:
        try:
            image_bytes = _img_to_bytes(imagem)
            with st.spinner("Extraindo matriz com Gemini..."):
                modelo_usado, matriz, raw = extrair_matriz_google_cached(image_bytes, n, n)
            st.success(f"Matriz extraída com {modelo_usado}.")
        except Exception as e:
            if _is_429(e):
                st.warning("Cota/limite da API (429). Use o preenchimento manual abaixo.")
            else:
                st.error(f"Falha ao extrair: {e}")

    texto_manual = st.text_area(
        "Cole a matriz manualmente (JSON ou linhas). Ex 4x4:\nA B C D\nE F G H\nI J K L\nM N O P",
        height=180,
    )

    if texto_manual.strip():
        m = parse_matriz_manual(texto_manual)
        if m is not None:
            try:
                matriz = ajustar_dimensoes(m, n, n)
                st.success("Matriz manual carregada.")
            except Exception as e:
                st.error(f"Matriz manual inválida: {e}")

    if matriz:
        st.write("Preview da matriz:")
        st.dataframe(pd.DataFrame(matriz), use_container_width=True)


# =========================================================
# SOLVER
# =========================================================
if matriz:
    trie_root, max_len_dict = carregar_trie_ptbr(MIN_PALAVRA, extras=frozenset(PALAVRAS_COMUNS_REF))

    t0 = time.perf_counter()
    achadas = buscar_palavras_boggle(matriz, trie_root, max_len_dict)
    dt = (time.perf_counter() - t0) * 1000

    palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))
    chave = escolher_palavra_chave(palavras, n)

    st.subheader("Resultado")
    st.write(f"- Palavras encontradas: **{len(palavras)}**")
    st.write(f"- Tempo do solver: **{dt:.1f} ms**")
    if chave:
        st.write(f"- Palavra-chave sugerida: **{chave}**")

    # agrupa por tamanho (mais legível)
    grupos = {}
    for w in palavras:
        grupos.setdefault(len(w), []).append(w)

    tamanhos = sorted(grupos.keys(), reverse=True)
    for L in tamanhos[:12]:  # evita UI gigante
        st.markdown(f"**{L} letras ({len(grupos[L])})**")
        st.write(", ".join(grupos[L][:120]))

    with st.expander("Exportar (CSV)"):
        df = pd.DataFrame(
            [{"palavra": w, "tamanho": len(w), "caminho": achadas[w]} for w in palavras]
        )
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Baixar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="palavras_boggle.csv",
            mime="text/csv",
        )
else:
    st.info("Envie uma imagem ou cole a matriz manualmente pra começar.")
