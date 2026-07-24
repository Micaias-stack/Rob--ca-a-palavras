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

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

# =========================================================
# STREAMLIT
# =========================================================
st.set_page_config(page_title="Robô Solver de Boggle", page_icon="🧩", layout="wide")
st.title("🧩 Robô Solver de Boggle")
st.caption("Upload da foto → IA extrai a grade → DFS acha palavras (PT-BR).")


# =========================================================
# CONTROLES: debounce / cooldown / espaçamento
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
    # lista simples PT-BR (github raw)
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
# OPENAI: extrair matriz via visão (com cache e retry)
# =========================================================
def extrair_retry_after_seconds(err: Exception) -> int | None:
    """
    Tenta ler "Retry-After" dos erros do SDK. Nem sempre vem.
    Se não vier, retorna None.
    """
    # openai errors geralmente têm .response (httpx Response) em alguns casos
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


def extrair_matriz_imagem_openai(
    imagem: Image.Image,
    api_key: str,
    n: int,
    model: str = "gpt-4o-mini",
):
    if not api_key:
        raise ValueError("Informe a OPENAI_API_KEY.")

    if "_matriz_cache" not in st.session_state:
        st.session_state["_matriz_cache"] = {}

    img_b64, img_hash = b64_e_hash(imagem)
    cache_key = f"{img_hash}:{n}:{model}"
    if cache_key in st.session_state["_matriz_cache"]:
        return st.session_state["_matriz_cache"][cache_key]

    client = OpenAI(api_key=api_key)

    prompt = (
        "Você está vendo uma foto de um tabuleiro estilo Boggle.\n"
        f"Extraia APENAS a matriz/grade completa de letras com tamanho {n}x{n}.\n"
        "Algumas células podem conter DUAS letras juntas (ex.: \"CH\", \"QU\"). "
        "Se acontecer, retorne essas duas letras na MESMA string.\n\n"
        "Responda EXCLUSIVAMENTE em JSON estrito (sem markdown, sem texto extra), no formato:\n"
        "{\"matriz\":[[\"E\",\"J\",\"S\",\"Z\"],[\"C\",\"CH\",\"I\",\"F\"],[\"A\",\"B\",\"O\",\"S\"],[\"R\",\"T\",\"U\",\"L\"]]}"
    )

    data_url = f"data:image/jpeg;base64,{img_b64}"

    last_err = None
    for attempt in range(6):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url, "detail": "high"},
                        ],
                    }
                ],
                max_output_tokens=250,
                temperature=0,
            )

            texto = getattr(resp, "output_text", "") or ""
            data = extrair_json_estrito(texto)
            matriz = normalizar_matriz(data.get("matriz"), n)

            st.session_state["_matriz_cache"][cache_key] = matriz
            return matriz

        except RateLimitError as e:
            last_err = e
            ra = extrair_retry_after_seconds(e)
            if ra is not None:
                set_cooldown(ra)
                raise ValueError(f"Rate limit: aguarde {ra}s e tente novamente.")
            # fallback exponencial
            time.sleep((2 ** attempt) + random.uniform(0, 1.0))

        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            time.sleep((2 ** attempt) + random.uniform(0, 1.0))

        except APIError as e:
            # 5xx/4xx gerais (não fica insistindo infinito)
            last_err = e
            time.sleep((2 ** attempt) + random.uniform(0, 1.0))

        except Exception as e:
            # erro de parsing / JSON / matriz inválida etc.
            last_err = e
            break

    raise ValueError(f"Falha ao extrair matriz via OpenAI: {last_err}")


# =========================================================
# UI
# =========================================================
with st.sidebar:
    st.subheader("Configuração")
    openai_key = st.text_input("OPENAI_API_KEY", type="password")
    tamanho = st.selectbox("Tamanho da grade", options=[4, 5], index=0)
    modelo = st.text_input("Modelo (visão)", value="gpt-4o-mini")
    st.caption("Dica: se bater rate limit, o app entra em cooldown automático.")


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    arquivo = st.file_uploader("Envie a foto do tabuleiro", type=["png", "jpg", "jpeg"])
    if arquivo:
        imagem = Image.open(arquivo)
        st.image(imagem, caption="Imagem enviada", use_container_width=True)

with col2:
    st.subheader("Resultado")
    if arquivo:
        if st.button("🚀 Resolver", type="primary"):
            debounce_click(5)
            check_cooldown()

            with st.spinner("Carregando dicionário PT-BR..."):
                dicionario, prefixos = carregar_dicionario_pt()

            with st.spinner("Extraindo matriz pela visão (OpenAI)..."):
                matriz = extrair_matriz_imagem_openai(
                    imagem=imagem,
                    api_key=openai_key,
                    n=int(tamanho),
                    model=modelo.strip(),
                )

            st.write("Matriz extraída:")
            st.dataframe(pd.DataFrame(matriz), use_container_width=True)

            with st.spinner("Buscando palavras (DFS)..."):
                achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)

            palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))
            df = pd.DataFrame(
                {
                    "palavra": palavras,
                    "tamanho": [len(p) for p in palavras],
                }
            )

            st.metric("Total de palavras", len(palavras))
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Baixar CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="boggle_palavras.csv",
                mime="text/csv",
            )
    else:
        st.info("Envie uma imagem para extrair a grade e resolver.")
