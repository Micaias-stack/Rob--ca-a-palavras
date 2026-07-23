import base64
import io
import json
import re
import urllib.request
import pandas as pd
import requests
from PIL import Image
import streamlit as st

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Robô Hacker de Boggle", page_icon="🧩", layout="wide"
)

st.title("🧩 Robô Solver de Boggle (Netflix)")
st.markdown(
    "Carregue a foto da TV. O robô vai ler a grade (inclusive blocos como CH) e calcular as maiores palavras do dicionário em zigue-zague!"
)

with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
    )
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]


# ==========================================
# MOTOR DICIONÁRIO PT-BR (CACHE)
# ==========================================
@st.cache_data
def carregar_dicionario_pt():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    try:
        resposta = urllib.request.urlopen(url)
        palavras_raw = resposta.read().decode("utf-8").splitlines()

        dicionario = set()
        prefixos = set()

        for p in palavras_raw:
            p = p.upper().strip()
            if len(p) >= 3 and "-" not in p and "." not in p:
                dicionario.add(p)
                for i in range(1, len(p) + 1):
                    prefixos.add(p[:i])

        return dicionario, prefixos
    except Exception as e:
        st.error(f"Erro ao baixar dicionário: {str(e)}")
        return set(), set()


# ==========================================
# MOTOR DE BUSCA ZIGUE-ZAGUE (DFS)
# ==========================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas > 0 else 0
    palavras_encontradas = {}

    def dfs(r, c, visitados, palavra_atual, caminho):
        letra_celula = str(matriz[r][c]).upper().strip()
        nova_palavra = palavra_atual + letra_celula

        if nova_palavra not in prefixos:
            return

        if nova_palavra in dicionario and len(nova_palavra) >= 3:
            if nova_palavra not in palavras_encontradas:
                palavras_encontradas[nova_palavra] = list(caminho)

        direcoes = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < linhas
                and 0 <= nc < colunas
                and (nr, nc) not in visitados
            ):
                visitados.add((nr, nc))
                dfs(nr, nc, visitados, nova_palavra, caminho + [(nr, nc)])
                visitados.remove((nr, nc))

    for r in range(linhas):
        for c in range(colunas):
            dfs(r, c, {(r, c)}, "", [(r, c)])

    return palavras_encontradas


# ==========================================
# CONVERSOR DE IMAGEM PARA BASE64
# ==========================================
def imagem_para_base64(imagem):
    buffered = io.BytesIO()
    imagem.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ==========================================
# EXTRAÇÃO VIA REST API DIRETA (À PROVA DE 404)
# ==========================================
def extrair_matriz_imagem(imagem, api_key):
    # 1. Consulta quais modelos estão liberados para a sua chave
    models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    res_models = requests.get(models_url)

    if res_models.status_code != 200:
        raise ValueError(
            f"Erro de autenticação na API (Código {res_models.status_code}): {res_models.text}"
        )

    models_data = res_models.json()
    modelos_disponiveis = [
        m["name"]
        for m in models_data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]

    if not modelos_disponiveis:
        raise ValueError(
            "Nenhum modelo compatível foi encontrado para esta API Key."
        )

    # Escolhe o modelo ativo
    modelo_escolhido = modelos_disponiveis[0]
    for pref in [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-2.0-flash",
    ]:
        if pref in modelos_disponiveis:
            modelo_escolhido = pref
            break

    # 2. Prepara e envia a requisição HTTP com a imagem
    img_b64 = imagem_para_base64(imagem)

    prompt = """
    Analise esta imagem de um jogo de palavras (Boggle).
    Extraia APENAS a matriz/grade completa de letras.
    MUITO IMPORTANTE: Observe se algumas células possuem DUAS letras juntas (exemplo: "CH", "QU", "RR"). 
    Extraia exatamente como está no bloco (mantenha o "CH" na mesma string da célula).

    Responda EXCLUSIVAMENTE em formato JSON estrito:
    {
      "matriz": [
        ["E", "J", "S", "Z", "Z", "S"],
        ["C", "CH", "I", "F", "O", "S"]
      ]
    }
    Não inclua markdown extra, apenas o JSON puro.
    """

    generate_url = f"https://generativelanguage.googleapis.com/v1beta/{modelo_escolhido}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64,
                        }
                    },
                ]
            }
        ]
    }

    response = requests.post(generate_url, json=payload)
    if response.status_code != 200:
        raise ValueError(
            f"Erro na requisição ({response.status_code}): {response.text}"
        )

    data = response.json()

    try:
        texto_resposta = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise ValueError(f"Resposta inesperada da API: {json.dumps(data)}")

    match_json = re.search(r"\{.*\}", texto_resposta, re.DOTALL)
    if match_json:
        dados = json.loads(match_json.group())
        return dados["matriz"]
    else:
        raise ValueError(
            "Falha ao ler o JSON da IA. Resposta recebida: " + texto_resposta
        )


# ==========================================
# FLUXO PRINCIPAL
# ==========================================
dicionario, prefixos = carregar_dicionario_pt()

uploaded_file = st.file_uploader(
    "Envie a foto da TV", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file and dicionario:
    imagem = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Imagem")
        st.image(imagem, use_column_width=True)

    with col2:
        st.subheader("⚙️ Status")

        if not api_key:
            st.warning("Insira sua Gemini API Key na barra lateral.")
        else:
            if st.button("🚀 Destruir no Boggle", type="primary"):
                with st.spinner("Conectando via REST API e lendo a grade..."):
                    try:
                        matriz = extrair_matriz_imagem(imagem, api_key)
                        st.success("Grade identificada!")

                        df_grid = pd.DataFrame(matriz)
                        st.dataframe(df_grid, use_container_width=True)

                        with st.spinner(
                            "Procurando combinações no dicionário..."
                        ):
                            resultados = buscar_palavras_boggle(
                                matriz, dicionario, prefixos
                            )

                        if resultados:
                            palavras_ordenadas = sorted(
                                resultados.keys(),
                                key=lambda x: len(x),
                                reverse=True,
                            )

                            st.subheader(
                                f"🔥 {len(palavras_ordenadas)} Palavras Encontradas!"
                            )
                            st.caption("As maiores palavras (mais pontos):")

                            for p in palavras_ordenadas[:50]:
                                st.markdown(f"- **{p}** ({len(p)} letras)")
                        else:
                            st.warning("Nenhuma palavra encontrada na grade.")

                    except Exception as e:
                        st.error(f"Erro no processamento: {str(e)}")
