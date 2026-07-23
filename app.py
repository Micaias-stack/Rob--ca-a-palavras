import json
import re
import google.generativeai as genai
import pandas as pd
from PIL import Image
import streamlit as st

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Robô Caça-Palavras com IA", page_icon="🧩", layout="wide"
)

st.title("🧩 Robô Solver de Caça-Palavras")
st.markdown(
    "Carregue a foto do seu caça-palavras e a IA identificará a grade e resolverá o jogo automaticamente!"
)

# Sidebar para gerenciamento da API Key
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Insira sua chave da API do Google Gemini.",
    )
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]


# ==========================================
# MOTOR DE BUSCA MATRICIAL (8 DIREÇÕES)
# ==========================================
def buscar_palavras(matriz, palavras):
    linhas = len(matriz)
    colunas = len(matriz[0]) if linhas > 0 else 0

    # Direções: (delta_linha, delta_coluna) -> 8 direções
    direcoes = [
        (-1, -1),
        (-1, 0),
        (-1, 1),  # Diagonais superiores e Cima
        (0, -1),
        (0, 1),  # Esquerda e Direita
        (1, -1),
        (1, 0),
        (1, 1),  # Diagonais inferiores e Baixo
    ]

    resultados = {}

    for palavra in palavras:
        palavra_limpa = palavra.upper().strip().replace(" ", "")
        tam = len(palavra_limpa)
        encontrada = False

        if not palavra_limpa:
            continue

        for r in range(linhas):
            if encontrada:
                break
            for c in range(colunas):
                if encontrada:
                    break

                # Se a primeira letra coincide, testa as 8 direções
                if matriz[r][c] == palavra_limpa[0]:
                    for dr, dc in direcoes:
                        coords = []
                        match = True

                        for i in range(tam):
                            nr, nc = r + dr * i, c + dc * i
                            if (
                                0 <= nr < linhas
                                and 0 <= nc < colunas
                                and matriz[nr][nc] == palavra_limpa[i]
                            ):
                                coords.append((nr, nc))
                            else:
                                match = False
                                break

                        if match:
                            resultados[palavra_limpa] = coords
                            encontrada = True
                            break

    return resultados


# ==========================================
# EXTRAÇÃO DE DADOS VIA IA (GEMINI VISION)
# ==========================================
def extrair_dados_imagem(imagem, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """
    Analise esta imagem de caça-palavras.
    Sua tarefa é extrair duas coisas:
    1. A matriz/grade completa de letras.
    2. A lista de palavras a serem encontradas no jogo (e regras relevantes, se houver).

    Responda EXCLUSIVAMENTE em formato JSON com a seguinte estrutura estrita:
    {
      "matriz": [
        ["A", "B", "C"],
        ["D", "E", "F"]
      ],
      "palavras": ["PALAVRA1", "PALAVRA2"]
    }
    
    Atenção:
    - Todas as letras da matriz e palavras devem estar em MAIÚSCULAS.
    - Garanta que todas as linhas da matriz tenham exatamente o mesmo número de colunas.
    - Não inclua markdown adicional além do bloco json.
    """

    response = model.generate_content([prompt, imagem])
    texto_resposta = response.text

    # Limpeza para garantir parsing de JSON
    match_json = re.search(r"\{.*\}", texto_resposta, re.DOTALL)
    if match_json:
        dados = json.loads(match_json.group())
        return dados["matriz"], dados["palavras"]
    else:
        raise ValueError(
            "Não foi possível estruturar os dados extraídos da imagem."
        )


# ==========================================
# FLUXO DA INTERFACE STREAMLIT
# ==========================================
uploaded_file = st.file_uploader(
    "Envie a foto do caça-palavras", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    imagem = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Imagem Enviada")
        st.image(imagem, use_column_width=True)

    with col2:
        st.subheader("⚙️ Processamento")

        if not api_key:
            st.warning(
                "Por favor, insira sua Gemini API Key na barra lateral para continuar."
            )
        else:
            if st.button("🚀 Resolver Caça-Palavras", type="primary"):
                with st.spinner(
                    "Lendo imagem e extraindo letras com IA..."
                ):
                    try:
                        matriz, palavras = extrair_dados_imagem(
                            imagem, api_key
                        )

                        st.success(
                            f"Matriz extraída: {len(matriz)}x{len(matriz[0])} | Palavras encontradas: {len(palavras)}"
                        )

                        # Executa algoritmo de busca
                        solucoes = buscar_palavras(matriz, palavras)

                        # Prepara coordenadas para destaque na interface
                        coords_destaque = set()
                        for coords in solucoes.values():
                            coords_destaque.update(coords)

                        # Renderização em formato tabular visual
                        st.subheader("🎯 Resultado")

                        def destacar_celulas(val, r, c):
                            if (r, c) in coords_destaque:
                                return "background-color: #28a745; color: white; font-weight: bold;"
                            return ""

                        df_grid = pd.DataFrame(matriz)

                        # Aplicação de estilo na grade
                        df_styled = df_grid.style.apply(
                            lambda x: [
                                (
                                    "background-color: #00e676; color: black; font-weight: bold;"
                                    if (r, c) in coords_destaque
                                    else "background-color: #f0f2f6; color: #333;"
                                )
                                for c in range(len(x))
                            ],
                            axis=1,
                        )

                        st.dataframe(df_grid, use_container_width=True)

                        st.markdown("### 📋 Status das Palavras")
                        for p in palavras:
                            p_clean = p.upper().strip()
                            if p_clean in solucoes:
                                st.markdown(f"- ✅ **{p_clean}** (Encontrada)")
                            else:
                                st.markdown(
                                    f"- ❌ **{p_clean}** (Não localizada na grade)"
                                )

                    except Exception as e:
                        st.error(
                            f"Ocorreu um erro durante o processamento: {str(e)}"
                        )
