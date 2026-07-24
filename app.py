import json
import re
import time
import traceback
import urllib.request
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# opcional (se instalado, melhora a escolha “rara” da palavra-chave)
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
# REGRAS / UPGRADES PEDIDOS
# =========================================================
TAMANHOS = {"4x4": 4, "5x5": 5, "6x6": 6}   # ✅ só 4/5/6
MIN_PALAVRA = 3                             # ✅ mínimo 3 letras para encontrar palavras

PONTOS_MAX_UNICA = 8                        # ✅ teto 8
PONTOS_REPETIDA = 1                         # repetida (2+ jogadores)
PONTOS_MAX_CHAVE = 8                        # palavra-chave até 8


# =========================================================
# DICIONÁRIO PT-BR (CACHE) - mantém o “jeito que tava”
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
# BOGGLE SOLVER (DFS) - upgrade: mínimo 3 letras
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
        letra = str(matriz[r][c]).upper().strip()
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
# GEMINI: Seleção robusta de modelo (mantém como tava)
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
    # ✅ “como tava”: pega do Streamlit secrets (sem input)
    # aceita qualquer um dos dois nomes, pra não quebrar seu deploy
    return (
        st.secrets.get("GOOGLE_API_KEY", "")
        or st.secrets.get("GEMINI_API_KEY", "")
    )


def extrair_matriz_google(imagem: Image.Image, linhas: int, colunas: int):
    api_key = _get_api_key()
    if not api_key:
        raise ValueError('API Key não configurada nos Secrets. Use st.secrets["GOOGLE_API_KEY"] ou st.secrets["GEMINI_API_KEY"].')

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

    try:
        m_ok = ajustar_dimensoes(m1, linhas, colunas)
        return modelo_id, m_ok, j1
    except Exception:
        pass

    # 2) retry: pede correção das dimensões
    resp2 = model.generate_content(
        [_prompt_correcao(linhas, colunas, m1), img_resized],
        request_options={"timeout": 120},
    )
    j2 = extrair_json_estrito(resp2.text)
    m2 = sanear_matriz(j2.get("matriz"))
    m_ok2 = ajustar_dimensoes(m2, linhas, colunas)

    return modelo_id, m_ok2, j2


# =========================================================
# UPGRADES: Palavra-chave + pontuação
# =========================================================
def _zipf_pt(word: str) -> float:
    if not WORDFREQ_OK:
        return 0.0
    return float(zipf_frequency((word or "").lower(), "pt"))


def min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)


def sugerir_palavra_chave(palavras: list[str], n_tabuleiro: int) -> str:
    """
    Escolha simples e estável:
    - tenta >= min_len por tabuleiro
    - desempate: maior tamanho
    - se wordfreq existir: mais rara (menor zipf)
    """
    if not palavras:
        return ""

    min_len = min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras if len(w) >= min_len]
    if not cand:
        maxlen = max(len(w) for w in palavras)
        cand = [w for w in palavras if len(w) == maxlen]

    if WORDFREQ_OK:
        # menor zipf = mais rara
        cand.sort(key=lambda w: (_zipf_pt(w), -len(w), w))
        return cand[0]

    # sem wordfreq: pega a mais longa; desempate alfabético
    cand.sort(key=lambda w: (-len(w), w))
    return cand[0]


def pontos_unica_por_tamanho(n: int) -> int:
    # cap em 8 (do jeito que você pediu)
    if n <= 2: return 0
    if n == 3: return 2
    if n == 4: return 3
    if n == 5: return 4
    if n == 6: return 6
    return PONTOS_MAX_UNICA  # 7+ => 8


def pontuar_palavra(palavra: str, qtd_jogadores_que_acharam: int, palavra_chave: str, pontos_chave: int) -> int:
    if palavra_chave and palavra == palavra_chave:
        return int(pontos_chave)
    if qtd_jogadores_que_acharam >= 2:
        return PONTOS_REPETIDA
    return pontos_unica_por_tamanho(len(palavra))


def parse_lista_palavras(txt: str) -> list[str]:
    # aceita separador por espaço, vírgula, quebra de linha
    raw = re.split(r"[\s,;]+", (txt or "").upper().strip())
    return [w for w in raw if w]


# =========================================================
# SIDEBAR (só 4x4/5x5/6x6)
# =========================================================
with st.sidebar:
    st.subheader("Grade")
    tamanho_label = st.selectbox("Tamanho do tabuleiro", list(TAMANHOS.keys()), index=2)
    n = TAMANHOS[tamanho_label]
    st.write(f"Mínimo de palavra: **{MIN_PALAVRA} letras**")

    st.divider()
    st.subheader("Pontuação (opcional)")
    pontos_chave = st.slider("Pontos da palavra-chave", 0, PONTOS_MAX_CHAVE, PONTOS_MAX_CHAVE, 1)
    qtd_jogadores = st.number_input("Qtd. jogadores", min_value=2, max_value=10, value=2, step=1)


# =========================================================
# MAIN: Upload + extração + solver
# =========================================================
colA, colB = st.columns([1.2, 1])

with colA:
    imagem_up = st.file_uploader("Envie a foto do tabuleiro", type=["png", "jpg", "jpeg", "webp"])

    if imagem_up:
        img = Image.open(imagem_up).convert("RGB")
        st.image(img, caption="Imagem enviada", use_container_width=True)

        if st.button("Extrair grade com Gemini", type="primary"):
            t0 = time.time()
            try:
                modelo_id, matriz, raw_json = extrair_matriz_google(img, n, n)
                st.session_state["matriz"] = matriz
                st.session_state["modelo_id"] = modelo_id
                st.session_state["raw_json"] = raw_json
                st.success(f"Grade extraída em {time.time()-t0:.1f}s (modelo: {modelo_id}).")
            except Exception as e:
                st.error(str(e))
                with st.expander("Detalhes do erro"):
                    st.code(traceback.format_exc(), language="text")

with colB:
    matriz = st.session_state.get("matriz")
    if matriz:
        st.subheader("Grade reconhecida")
        st.dataframe(pd.DataFrame(matriz), use_container_width=True)

        with st.expander("JSON bruto retornado pelo modelo"):
            st.json(st.session_state.get("raw_json", {}))


# =========================================================
# Solver + Palavra-chave + listagem
# =========================================================
matriz = st.session_state.get("matriz")
if matriz:
    dicionario, prefixos = carregar_dicionario_pt()

    st.divider()
    st.subheader("Palavras encontradas")

    achadas = buscar_palavras_boggle(matriz, dicionario, prefixos, min_palavra=MIN_PALAVRA)
    palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))

    palavra_chave = sugerir_palavra_chave(palavras, n)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Total", len(palavras))
    c2.metric("Palavra-chave", palavra_chave or "—")
    c3.caption("Dica: se quiser uma palavra-chave mais “rara”, instale `wordfreq` no deploy.")

    max_exibir = st.slider("Máximo para exibir", 50, 2000, 300, 50)
    st.write(", ".join(palavras[:max_exibir]) if palavras else "Nenhuma palavra encontrada.")

    # =========================================================
    # Pontuação multi-jogadores (opcional)
    # =========================================================
    st.divider()
    st.subheader("Pontuação (multi-jogadores)")

    st.caption(
        "Regras: palavra repetida (2+ jogadores) = 1 ponto; palavra única = por tamanho (cap 8); "
        f"palavra-chave = {pontos_chave} pontos."
    )

    validas_set = set(palavras)
    entradas = []
    cols = st.columns(int(qtd_jogadores))
    for i in range(int(qtd_jogadores)):
        with cols[i]:
            txt = st.text_area(f"Jogador {i+1} (cole as palavras)", height=140, key=f"jog_{i}")
            lst = parse_lista_palavras(txt)
            entradas.append(lst)

    # valida e cria sets por jogador (somente palavras do solver)
    sets_validas = []
    invalidas_por_jogador = []
    for lst in entradas:
        validas = [w for w in lst if w in validas_set]
        invalidas = [w for w in lst if w and w not in validas_set]
        sets_validas.append(set(validas))
        invalidas_por_jogador.append(sorted(set(invalidas)))

    # contagem global: quantos jogadores acharam cada palavra
    contagem = Counter()
    for s in sets_validas:
        contagem.update(s)

    # calcula pontos por jogador
    linhas_tabela = []
    for idx, s in enumerate(sets_validas):
        total = 0
        detalhes = []
        for w in sorted(s, key=lambda x: (-len(x), x)):
            p = pontuar_palavra(w, contagem[w], palavra_chave, pontos_chave)
            total += p
            detalhes.append((w, p, contagem[w]))
        linhas_tabela.append((idx + 1, total, detalhes))

    # output
    resumo = pd.DataFrame(
        [{"Jogador": f"Jogador {j}", "Pontos": pts} for j, pts, _ in linhas_tabela]
    ).sort_values("Pontos", ascending=False)

    st.dataframe(resumo, use_container_width=True, hide_index=True)

    with st.expander("Detalhamento por jogador"):
        for j, pts, detalhes in linhas_tabela:
            st.markdown(f"**Jogador {j} — {pts} pontos**")
            if not detalhes:
                st.write("—")
            else:
                df_det = pd.DataFrame(detalhes, columns=["Palavra", "Pontos", "Qtd. jogadores"])
                st.dataframe(df_det, use_container_width=True, hide_index=True)

    with st.expander("Palavras inválidas (não encontradas pelo robô)"):
        for i, inv in enumerate(invalidas_por_jogador):
            st.markdown(f"**Jogador {i+1}:** {', '.join(inv) if inv else '—'}")
