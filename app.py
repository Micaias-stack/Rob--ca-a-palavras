# app.py
import json
import re
import time
import urllib.request
import unicodedata
from collections import defaultdict

import pandas as pd
import streamlit as st
from PIL import Image

import google.generativeai as genai

# opcional (melhor filtro de “palavras do dia a dia”)
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
# NORMALIZAÇÃO (acentos)
# =========================================================
def strip_accents(s: str) -> str:
    s = s or ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def canon_word(s: str) -> str:
    # canonical para busca: SEM ACENTO, MAIÚSCULO, só letras
    s = strip_accents(s).upper().strip()
    s = re.sub(r"[^A-Z]", "", s)
    return s


# =========================================================
# DICIONÁRIO PT-BR (CACHE) — agora canônico (sem acento)
# =========================================================
@st.cache_data(ttl=21600)  # 6h
def carregar_dicionario_pt_canon():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    with urllib.request.urlopen(url, timeout=60) as resposta:
        palavras_raw = resposta.read().decode("utf-8", errors="ignore").splitlines()

    dicionario_canon = set()
    prefixos_canon = set()
    canon_to_forms = defaultdict(set)

    for p in palavras_raw:
        p0 = (p or "").strip()
        if not p0:
            continue

        # rejeita coisas claramente inválidas p/ jogo (abreviações com ponto etc.)
        if any(ch.isdigit() for ch in p0) or "." in p0 or "-" in p0 or "'" in p0:
            continue

        c = canon_word(p0)
        if len(c) < 2:
            continue

        dicionario_canon.add(c)
        canon_to_forms[c].add(p0.strip())

        for i in range(1, len(c) + 1):
            prefixos_canon.add(c[:i])

    return dicionario_canon, prefixos_canon, dict(canon_to_forms)


# =========================================================
# PONTUAÇÃO (Boggle clássico)
# Dígrafos QU/RR/CH já contam como 2 porque len("QU")==2 etc.
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
# FILTROS (regras do jogo)
# =========================================================
SUFIXOS_DIMINUTIVO = (
    "ZINHO", "ZINHA", "INHO", "INHA", "ZITO", "ZITA",
)
SUFIXOS_AUMENTATIVO = (
    "ZAO", "ZONA", "AÇO",  # "AÇO" vai virar "ACO" no canônico
)

# lista básica (bem conservadora). Você pode colar mais no app.
OFENSIVAS_BASE_CANON = {
    # deixe essa lista curta/segura; o ideal é você completar por fora
    "PORRA",
    "CARALHO",
    "PUTA",
    "PUTO",
    "MERDA",
    "BOSTA",
    "FODER",
    "FODA",
}

def tem_sufixo_dim_aum(canon: str) -> bool:
    c = canon or ""
    if any(c.endswith(suf) for suf in SUFIXOS_DIMINUTIVO):
        return True
    # aumentativos: cuidado pra não bloquear muita palavra comum, então fica bem restrito
    if c.endswith("ZAO") or c.endswith("ZONA"):
        return True
    return False


def zipf_pt(palavra_com_acento: str) -> float:
    if not WORDFREQ_OK:
        return 0.0
    return float(zipf_frequency((palavra_com_acento or "").lower(), "pt"))


def escolher_melhor_forma(canon: str, canon_to_forms: dict) -> str:
    forms = list(canon_to_forms.get(canon, []))
    if not forms:
        return canon

    if WORDFREQ_OK:
        # pega a forma mais provável no português (tende a “colocar acento” certo)
        forms.sort(key=lambda w: zipf_pt(w), reverse=True)
        return forms[0]

    # fallback: primeira (ordem alfabética)
    return sorted(forms)[0]


def parse_blacklist(texto: str) -> set:
    itens = set()
    for linha in (texto or "").splitlines():
        t = linha.strip()
        if not t:
            continue
        itens.add(canon_word(t))
    return itens


def palavra_do_cotidiano(form: str, min_zipf: float) -> bool:
    # “comum e não tão comum usada no cotidiano”
    if not WORDFREQ_OK:
        return True
    return zipf_pt(form) >= float(min_zipf)


# =========================================================
# BOGGLE SOLVER (DFS) — usa CANÔNICO (sem acento)
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0

    achadas = {}  # canon -> caminho
    direcoes = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def dfs(r, c, visitados, palavra, caminho):
        letra = canon_word(str(matriz[r][c]))
        nova = palavra + letra

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
# GEMINI: JSON & MATRIX
# =========================================================
def extrair_json_estrito(texto: str) -> dict:
    t = (texto or "").strip()

    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"JSON não encontrado na resposta do modelo.\nResposta (início): {t[:400]}")
    return json.loads(m.group(0))


def _cell_ok(x: str) -> bool:
    s = (x or "").strip()
    return bool(re.search(r"[A-Za-zÀ-Ü]", s))


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

    raise RuntimeError("Nenhum modelo com suporte a generateContent foi encontrado no seu projeto.")


def _prompt_extracao(linhas, colunas):
    return (
        f"Você receberá a imagem de um tabuleiro tipo Boggle com {linhas} linhas e {colunas} colunas.\n"
        "Extraia as letras de cada célula.\n"
        "Regras:\n"
        "- Leia da esquerda para a direita, de cima para baixo.\n"
        "- Se uma célula tiver múltiplas letras (ex: 'QU', 'RR', 'CH'), mantenha junto.\n"
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

    # 1) tentativa principal
    response = model.generate_content([_prompt_extracao(linhas, colunas), img_resized], request_options={"timeout": 120})
    j = extrair_json_estrito(response.text)
    matriz = sanear_matriz(j.get("matriz"))

    # 2) valida / ajusta, senão retry
    try:
        matriz_ok = ajustar_dimensoes(matriz, linhas, colunas)
        return modelo_id, {"matriz": matriz_ok}
    except Exception:
        response2 = model.generate_content(_prompt_correcao(linhas, colunas, matriz), request_options={"timeout": 120})
        j2 = extrair_json_estrito(response2.text)
        matriz2 = sanear_matriz(j2.get("matriz"))
        matriz_ok2 = ajustar_dimensoes(matriz2, linhas, colunas)
        return modelo_id, {"matriz": matriz_ok2}


# =========================================================
# MAIN APP
# =========================================================
dicionario_canon, prefixos_canon, canon_to_forms = carregar_dicionario_pt_canon()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1) Envie a foto")
    uploaded_file = st.file_uploader("Selecione a imagem (.jpg, .png)", type=["jpg", "png", "jpeg"])

    opcoes_tamanho = ["4x4", "5x5", "5x4", "6x4", "5x6", "6x5", "6x6"]
    tamanho_selecionado = st.selectbox("Tamanho do tabuleiro (Linhas x Colunas):", opcoes_tamanho, index=len(opcoes_tamanho) - 1)

    st.subheader("2) Filtros (regras do jogo)")
    min_len = st.slider("Tamanho mínimo da palavra", 3, 8, 4, 1)
    qtd_palavras = st.slider("Quantas palavras mostrar?", 10, 500, 120, 10)

    bloquear_dim_aum = st.checkbox("Bloquear diminutivos/aumentativos", value=True)
    bloquear_ofensivas = st.checkbox("Bloquear palavras muito ofensivas", value=True)

    usar_cotidiano = st.checkbox("Só palavras do cotidiano (recomendado)", value=True, disabled=not WORDFREQ_OK)
    min_zipf = st.slider("Nível de 'cotidiano' (maior = mais comum)", 2.5, 5.0, 3.7, 0.1, disabled=not WORDFREQ_OK)

    st.caption("Extras: você pode colar palavras para bloquear (nomes próprios, marcas, lugares, etc.).")
    blacklist_text = st.text_area("Blacklist (1 por linha)", height=120, placeholder="Ex:\nMARIA\nPEPSI\nURUGUAI\nGOOGLE")

    if not WORDFREQ_OK:
        st.info("Para o filtro 'cotidiano', adicione `wordfreq` no requirements.txt.")

    if uploaded_file:
        imagem = Image.open(uploaded_file)
        st.image(imagem, caption="Imagem carregada", use_container_width=True)

with col2:
    st.subheader("Resultado")

    if uploaded_file and st.button(f"Analisar tabuleiro {tamanho_selecionado}", use_container_width=True):
        linhas, colunas = map(int, tamanho_selecionado.split("x"))
        blacklist = parse_blacklist(blacklist_text)
        ofensivas = set(OFENSIVAS_BASE_CANON)

        with st.spinner("🔍 Extraindo a matriz com Gemini..."):
            t0 = time.time()
            modelo_usado, json_resposta = extrair_matriz_google(GOOGLE_API_KEY, imagem, linhas, colunas)
            matriz = json_resposta["matriz"]
            t1 = time.time()

        st.success(f"Matriz {linhas}x{colunas} extraída em {t1 - t0:.1f}s.")
        st.info(f"Modelo Gemini em uso: {modelo_usado}")
        st.code(json.dumps(matriz, indent=2, ensure_ascii=False), language="json")

        with st.spinner("🧠 Buscando palavras..."):
            t2 = time.time()
            achadas_canon = buscar_palavras_boggle(matriz, dicionario_canon, prefixos_canon)
            t3 = time.time()

        # aplica regras / filtros
        filtradas = {}
        for canon, caminho in achadas_canon.items():
            tam = len(canon)
            if tam < int(min_len):
                continue

            if canon in blacklist:
                continue

            if bloquear_ofensivas and canon in ofensivas:
                continue

            if bloquear_dim_aum and tem_sufixo_dim_aum(canon):
                continue

            # escolhe forma “bonita” (com acento, se houver)
            forma = escolher_melhor_forma(canon, canon_to_forms)

            # “cotidiano”
            if usar_cotidiano and WORDFREQ_OK:
                if not palavra_do_cotidiano(forma, min_zipf):
                    continue

            filtradas[canon] = {"forma": forma, "caminho": caminho, "tamanho": tam, "pontos": pontos_boggle(tam)}

        if not filtradas:
            st.warning("Nenhuma palavra encontrada com esses filtros.")
        else:
            # ordena por pontos, tamanho e frequência (se tiver wordfreq)
            itens = list(filtradas.items())

            def chave(item):
                canon, data = item
                freq = zipf_pt(data["forma"]) if WORDFREQ_OK else 0.0
                return (data["pontos"], data["tamanho"], freq, data["forma"])

            itens.sort(key=chave, reverse=True)
            itens = itens[: int(qtd_palavras)]

            df = pd.DataFrame(
                [{
                    "palavra": data["forma"],
                    "tamanho": data["tamanho"],
                    "pontos": data["pontos"],
                } for _, data in itens]
            )

            st.success(f"Mostrando {len(df)} palavras (em {t3 - t2:.1f}s).")
            st.metric("Pontos totais (somente as exibidas)", int(df["pontos"].sum()))
            st.dataframe(df, use_container_width=True)
