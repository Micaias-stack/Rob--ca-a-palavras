import base64
import hashlib
import io
import json
import random
import re
import time
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image

from groq import Groq
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

try:
    from paddleocr import PaddleOCR
    OCR_DISPONIVEL = True
except ImportError:
    OCR_DISPONIVEL = False

# =========================================================
# STREAMLIT
# =========================================================
st.set_page_config(page_title="Robô Solver de Boggle", page_icon="🧩", layout="wide")
st.title("🧩 Robô Solver de Boggle (OCR + Groq)")
st.caption("Upload da foto → OCR extrai a grade → Groq valida → DFS acha palavras (PT-BR).")

# =========================================================
# CONTROLES: debounce / cooldown
# =========================================================
def debounce_click(intervalo: float = 5.0):
    agora = time.time()
    ultimo = st.session_state.get("_last_click_ts", 0.0)
    if agora - ultimo < intervalo:
        st.warning(f"Aguarde {(intervalo - (agora - ultimo)):.1f}s e tente novamente.")
        st.stop()
    st.session_state["_last_click_ts"] = agora


def set_cooldown(segundos: int):
    st.session_state["_cooldown_until"] = time.time() + max(0, int(segundos))


def check_cooldown():
    until = st.session_state.get("_cooldown_until", 0.0)
    if time.time() < until:
        faltam = until - time.time()
        st.warning(f"Rate limit/limite temporário. Aguarde {faltam:.0f}s e tente novamente.")
        st.stop()


# =========================================================
# DICIONÁRIO PT-BR (CACHE)
# =========================================================
@st.cache_data(ttl=21600)  # 6h
def carregar_dicionario_pt():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    resposta = urllib.request.urlopen(url, timeout=60)
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
# BOGGLE DFS
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas else 0
    achadas = {}

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
                visitados.add((nr, nc))
                dfs(nr, nc, visitados, nova, caminho + [(nr, nc)])
                visitados.remove((nr, nc))

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return achadas


# =========================================================
# IMAGEM -> B64 + HASH (cache por foto)
# =========================================================
def b64_e_hash(imagem: Image.Image):
    img = imagem.copy().convert("RGB")
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("utf-8"), hashlib.sha256(raw).hexdigest()


# =========================================================
# UTIL: extrair JSON estrito
# =========================================================
def extrair_json_estrito(texto: str) -> dict:
    texto = (texto or "").strip()

    # 1) JSON puro
    try:
        return json.loads(texto)
    except Exception:
        pass

    # 2) Primeiro bloco {...}
    m = re.search(r"\{[\s\S]*\}", texto)
    if not m:
        raise ValueError(f"Não consegui encontrar JSON no retorno do modelo.\nRetorno:\n{texto[:2000]}")
    return json.loads(m.group(0))


def normalizar_matriz(matriz, n):
    if not isinstance(matriz, list) or len(matriz) != n:
        raise ValueError(f"Matriz inválida: esperado {n} linhas.")
    out = []
    for i, row in enumerate(matriz):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"Matriz inválida: linha {i} deveria ter {n} colunas.")
        out.append([str(x).upper().strip() for x in row])
    return out


# =========================================================
# OCR: extrair texto da imagem (PaddleOCR)
# =========================================================
@st.cache_resource
def get_ocr_engine():
    if not OCR_DISPONIVEL:
        return None
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def extrair_texto_ocr(imagem: Image.Image) -> str:
    ocr = get_ocr_engine()
    if ocr is None:
        raise ValueError("PaddleOCR não instalado. Instale com: pip install paddleocr paddlepaddle")
    
    img = imagem.copy().convert("RGB")
    img.thumbnail((1200, 1200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    
    result = ocr.ocr(buf.read(), cls=True)
    
    if not result or not result[0]:
        return ""
    
    # Ordena por posição (top-left → bottom-right) e concatena
    boxes = []
    for line in result[0]:
        box, (text, conf) = line
        y_center = (box[0][1] + box[2][1]) / 2
        x_center = (box[0][0] + box[2][0]) / 2
        boxes.append((y_center, x_center, text))
    
    boxes.sort()
    return " ".join([text for _, _, text in boxes])


# =========================================================
# GROQ: validar/corrigir matriz (sem visão, só texto)
# =========================================================
def extrair_retry_after_seconds(err: Exception) -> int | None:
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    try:
        headers = resp.headers
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra and str(ra).strip().isdigit():
            return int(str(ra).strip())
    except Exception:
        return None
    return None


def validar_matriz_groq(
    texto_ocr: str,
    api_key: str,
    n: int,
    model: str = "llama-3.3-70b-versatile",
):
    if not api_key:
        raise ValueError("Informe a GROQ_API_KEY.")

    client = Groq(api_key=api_key)

    prompt = (
        f"Você recebeu o texto extraído por OCR de um tabuleiro Boggle {n}x{n}.\n"
        f"Texto OCR: \"{texto_ocr}\"\n\n"
        f"Sua tarefa:\n"
        f"1) Identifique APENAS as {n*n} letras do tabuleiro (ignorando números, pontos, ruído).\n"
        f"2) Organize essas letras numa matriz {n}x{n} (leia linha por linha, esquerda→direita, cima→baixo).\n"
        f"3) Algumas células podem ter DUAS letras juntas (ex.: \"CH\", \"QU\"). Se acontecer, coloque as duas letras na mesma string.\n"
        f"4) Retorne EXCLUSIVAMENTE em JSON estrito (sem markdown, sem texto extra):\n"
        f"{{\"matriz\":[[\"A\",\"B\",\"C\",\"D\"],[\"E\",\"F\",\"G\",\"H\"],[\"I\",\"J\",\"K\",\"L\"],[\"M\",\"N\",\"O\",\"P\"]]}}"
    )

    last_err = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )

            texto = resp.choices[0].message.content or ""
            data = extrair_json_estrito(texto)
            matriz = normalizar_matriz(data.get("matriz"), n)
            return matriz

        except RateLimitError as e:
            last_err = e
            ra = extrair_retry_after_seconds(e)
            if ra is not None:
                set_cooldown(ra)
                raise ValueError(f"Rate limit atingido. Aguarde {ra}s.")
            time.sleep((2 ** attempt) + random.uniform(0, 1.0))

        except (APIConnectionError, APITimeoutError, APIError) as e:
            last_err = e
            time.sleep((2 ** attempt) + random.uniform(0, 1.0))

        except Exception as e:
            raise ValueError(f"Erro ao validar matriz com Groq: {e}")

    raise ValueError(f"Falha ao validar matriz com Groq após retries: {last_err}")


# =========================================================
# PIPELINE COMPLETO: OCR → Groq → Matriz
# =========================================================
def extrair_matriz_imagem_ocr_groq(
    imagem: Image.Image,
    groq_key: str,
    n: int,
):
    if "_matriz_cache" not in st.session_state:
        st.session_state["_matriz_cache"] = {}

    _, img_hash = b64_e_hash(imagem)
    cache_key = f"{img_hash}:{n}:ocr_groq"
    if cache_key in st.session_state["_matriz_cache"]:
        return st.session_state["_matriz_cache"][cache_key]

    # 1) OCR extrai texto
    with st.spinner("Extraindo texto da imagem (OCR)..."):
        texto_ocr = extrair_texto_ocr(imagem)
        if not texto_ocr:
            raise ValueError("OCR não conseguiu extrair texto da imagem. Tente uma foto com melhor iluminação/ângulo.")

    st.info(f"**Texto extraído pelo OCR:** {texto_ocr[:300]}...")

    # 2) Groq organiza em matriz
    with st.spinner("Validando matriz com Groq..."):
        matriz = validar_matriz_groq(texto_ocr, groq_key, n)

    st.session_state["_matriz_cache"][cache_key] = matriz
    return matriz


# =========================================================
# UI: upload + processamento
# =========================================================
st.subheader("📤 Upload da foto do tabuleiro")

if not OCR_DISPONIVEL:
    st.error("⚠️ PaddleOCR não instalado. Instale com: `pip install paddleocr paddlepaddle`")
    st.stop()

groq_key = st.text_input("**GROQ_API_KEY**", type="password", help="Pegue em: https://console.groq.com/keys")
tamanho_grade = st.selectbox("**Tamanho da grade**", [4, 5], index=0, help="4x4 ou 5x5")

uploaded_file = st.file_uploader("Escolha a imagem do Boggle", type=["jpg", "jpeg", "png"])

if uploaded_file:
    imagem = Image.open(uploaded_file)
    st.image(imagem, caption="Imagem carregada", use_container_width=True)

    if st.button("🚀 Extrair matriz e resolver", type="primary"):
        check_cooldown()
        debounce_click()

        if not groq_key:
            st.error("Informe a GROQ_API_KEY.")
            st.stop()

        try:
            matriz = extrair_matriz_imagem_ocr_groq(imagem, groq_key, tamanho_grade)

            st.success("✅ Matriz extraída com sucesso!")
            st.json({"matriz": matriz})

            with st.spinner("Carregando dicionário PT-BR..."):
                dicionario, prefixos = carregar_dicionario_pt()

            with st.spinner("Buscando palavras válidas..."):
                palavras = buscar_palavras_boggle(matriz, dicionario, prefixos)

            if not palavras:
                st.warning("Nenhuma palavra válida encontrada.")
            else:
                df = pd.DataFrame([
                    {"Palavra": p, "Tamanho": len(p), "Caminho": " → ".join([f"({r},{c})" for r, c in cam])}
                    for p, cam in palavras.items()
                ]).sort_values(by="Tamanho", ascending=False).reset_index(drop=True)

                st.success(f"🎯 {len(palavras)} palavras encontradas!")
                st.dataframe(df, use_container_width=True, height=400)

        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")
