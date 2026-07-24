import json
import re
import time
import traceback
import urllib.request

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
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Robô Caça-Palavras (Boggle)", page_icon="🧩", layout="wide")
st.title("🧩 Robô Caça-Palavras (Gemini + Python)")
st.caption("Upload da foto → Gemini extrai a grade → Python encontra palavras (PT-BR).")


# =========================================================
# REGRAS / MELHORIAS
# =========================================================
TAMANHOS = {"4x4": 4, "5x5": 5, "6x6": 6}  # apenas 4/5/6
MIN_PALAVRA = 3                            # ✅ mínimo 3 letras

# Palavra-chave: regras por tamanho do tabuleiro
def _min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def escolher_palavra_chave(palavras_encontradas: list[str], n_tabuleiro: int) -> str:
    """
    Palavra-chave:
    - prioriza palavras longas (por tabuleiro)
    - se wordfreq existir: tenta "cotidiano mas não muito comum"
      (faixa Zipf ~ 3.0 a 4.6, alvo 3.8)
    """
    if not palavras_encontradas:
        return ""

    min_len = _min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras_encontradas if len(w) >= min_len]

    if not cand:
        maxlen = max(len(w) for w in palavras_encontradas)
        cand = [w for w in palavras_encontradas if len(w) == maxlen]

    if WORDFREQ_OK:
        def z(w: str) -> float:
            return float(zipf_frequency(w.lower(), "pt"))

        # "cotidiano mas não comum"
        cand_mid = [w for w in cand if 3.0 <= z(w) <= 4.6]
        if cand_mid:
            cand = cand_mid

        alvo = 3.8
        def score(w: str):
            return (len(w), -abs(z(w) - alvo))

        return max(cand, key=score)

    # sem wordfreq: pega a maior (estável)
    cand.sort(key=lambda w: (-len(w), w))
    return cand[0]


# =========================================================
# DICIONÁRIO PT-BR (CACHE) + PALAVRAS COMUNS DE REFERÊNCIA
# =========================================================
# ✅ Referência: palavras comuns aceitas (você enviou)
# Agora o código tem noção destas palavras ao escolher palavra-chave
PALAVRAS_COMUNS_REF = {
    # 3 letras
    "QUE", "COM", "POR", "UMA", "NAS", "NOS", "DAS", "DOS", "SEM", "AOS", "MAS", "PRA", "PRO", "TEM", "FOI", 
    "SOU", "ERA", "SER", "VER", "SEU", "SUA", "MEU", "TEU", "TUA", "ELE", "ELA", "NÓS", "NÃO", "SIM", "NEM", 
    "DEZ", "MIL", "CEM", "VEZ", "ANO", "DIA", "BEM", "MAL", "BOM", "BOA", "DOR", "COR", "LUZ", "SOL", "PAZ", 
    "FEZ", "VAI", "VEM", "VOU", "USA", "USO", "SOB", "LHE", "TAL", "NUM", "UNS", "MIM", "ALI", "ORA", "OCO", 
    "TÃO", "MÃO", "PÃO", "SÃO", "DAR", "LER", "RIR", "PÔR", "TIA", "TIO", "PAI", "MÃE", "AVÓ", "AVÔ", "FAZ", 
    "DIZ", "LEI", "LUA", "MAR", "VIA", "RUA", "VOZ", "SOM", "SAI", "VIU", "DEU", "CAI", "PÔS", "PÕE", "ATO", 
    "EIS", "CAL", "MEL", "SAL", "CHÁ", "CÉU", "OLÁ", "ECO", "GÁS", "SUL",
    
    # 4 letras
    "PARA", "MAIS", "COMO", "ONDE", "AQUI", "HOJE", "CASO", "POIS", "PELA", "PELO", "ESTE", "ESTA", "ISTO", 
    "ISSO", "DELE", "DELA", "NELE", "NELA", "CADA", "ALGO", "NADA", "TUDO", "VIDA", "CASA", "LADO", "MODO", 
    "HORA", "DIAS", "ANOS", "BENS", "RUAS", "RUIM", "ALTO", "LEVE", "DOCE", "PODE", "QUER", "FALA", "FICA", 
    "VEJA", "VOCÊ", "ELES", "ELAS", "DOIS", "TRÊS", "SETE", "OITO", "NOVE", "DOZE", "LOGO", "CEDO", "SAIR", 
    "SAIU", "VEIO", "COME", "COMI", "BEBE", "BEBI", "LEVA", "PAGO", "PAGA", "ABRE", "ABRI", "AMEI", "MORA", 
    "MORO", "CALA", "CALE", "RIEM", "RISO", "VEEM", "PÔDE", "APÓS", "AZUL", "ROSA", "ROXO", "BEGE", "FLOR", 
    "FRIO", "NEVE", "MARÉ", "LAGO", "MATA", "MESA", "CAMA", "SOFÁ", "SACO", "VELA", "FOGO", "TELA", "PANO", 
    "POTE", "TUBO", "PEÇA", "LEIS", "REIS", "MÃES",
    
    # 5 letras
    "MUITO", "ASSIM", "AINDA", "POUCO", "TEMPO", "GENTE", "COISA", "PARTE", "VEZES", "NUNCA", "SOBRE", "FAZER", 
    "PODER", "DEVER", "DEIXA", "DESSA", "DESSE", "DESTE", "MESMO", "MENOS", "AGORA", "ANTES", "NOITE", "TARDE", 
    "CERTO", "SERIA", "SABER", "ACHAR", "FICAR", "LEVAR", "PEDIR", "DIZER", "FALAR", "VIVER", "MUNDO", "LIVRO", 
    "JOVEM", "IDOSO", "HOMEM", "CARRO", "CASAS", "RUINS", "ALTOS", "BAIXO", "LEVES", "DOCES", "CLARO", "PRETO", 
    "VERDE", "AZUIS", "CINZA", "FELIZ", "CHATO", "CHEIO", "VAZIO", "FORTE", "FRACO", "CURTO", "LONGO", "OUTRO", 
    "MESMA", "TODOS", "TODAS", "MAIOR", "MENOR", "DEMAIS", "ACIMA", "ENTRE", "DESDE", "ONTEM", "CEDER", "SAÍDA", 
    "CHEGA", "FEITA", "FEITO", "DISSE", "FOMOS", "FORAM", "ESTAR", "ESTOU", "ESTÃO", "SERÃO", "TERIA", "HOUVE", 
    "HAVER", "POSSO", "POSSA", "SABEM", "QUERO", "VAMOS", "VIMOS", "BUSCA", "ACHOU", "TRAGO", "LEITO", "LEMAS", 
    "TEMAS", "ITENS", "DADOS", "FATOS",
    
    # 6 letras
    "PORQUE", "DEPOIS", "EMBORA", "APENAS", "SEMPRE", "JAMAIS", "DENTRO", "ABAIXO", "FRENTE", "APESAR", "PESSOA", 
    "ESCOLA", "AMIGOS", "AMIGAS", "COMIDA", "BEBIDA", "SALADA", "CIDADE", "ESTADO", "BAIRRO", "CENTRO", "PLANTA", 
    "ANIMAL", "IMAGEM", "TAREFA", "ESTUDO", "MATRIZ", "PLANOS", "TOMADA", "SEGURO", "MÍNIMO", "MÁXIMO", "RÁPIDO", 
    "MODELO", "MÉTODO", "TEORIA", "CÓDIGO", "TÓPICO", "PÁGINA", "TÍTULO", "LISTAS", "TEXTOS", "FRASES", "ACESSO", 
    "SENHAS", "MOEDAS", "PREÇOS", "CUSTOS", "VENDAS", "COMPRA", "BOLETO", "CARTÃO", "DÉBITO", "EQUIPE", "CORPOS", 
    "CABEÇA", "BRAÇOS", "PERNAS", "LÁBIOS", "CABELO", "COSTAS", "OUVIDO", "VACINA", "DOENÇA", "DOENTE", "FEBRES", 
    "CÂNCER", "MEDIDA", "QUILOS", "LITROS", "METROS", "GRAMAS", "GRANDE", "PRONTO", "PRONTA", "QUENTE", "ESCURO", 
    "GELADO", "ALEGRE", "TRISTE", "JUSTOS", "CERTOS", "ERRADO", "ERRADA", "BRANCO", "CINZAS", "AMANHÃ", "QUARTA", 
    "QUINTA", "SÁBADO", "TERÇAS", "CORRER", "VENDER", "BUSCAR", "CHEGAR", "VOLTAR", "FECHAR", "PASSAR", "OBJETO", 
    "MOTIVO",
    
    # 7 letras
    "EXEMPLO", "PESSOAS", "FAMÍLIA", "CRIANÇA", "PRECISO", "PRECISA", "CAMINHO", "MOMENTO", "SEGUNDO", "MERCADO", 
    "SISTEMA", "PRODUTO", "SERVIÇO", "CUIDADO", "CELULAR", "PROJETO", "REUNIÃO", "VIAGENS", "DOMINGO", "SEGUNDA", 
    "LIMPEZA", "COZINHA", "CADEIRA", "TIJOLOS", "AMARELO", "ALEGRIA", "OBJETOS", "GARRAFA", "SAPATOS", "CAMISAS", 
    "ARMÁRIO", "CADERNO", "LEITURA", "SORRISO", "PINTURA", "RETRATO", "DESENHO", "CLIENTE", "EMPRESA", "NEGÓCIO", 
    "ESTADOS", "CENTROS", "ESTRADA", "AVENIDA", "RODOVIA", "VIADUTO", "PEQUENO", "PEQUENA", "MAIORIA", "MENINOS", 
    "MENINAS", "ABRINDO", "PAGANDO", "APLICAR", "ACEITAR", "AJUDADO", "CANSADO", "CANSADA", "FELIZES", "TRISTES", 
    "CERTEZA", "DÚVIDAS", "MATÉRIA", "TECLADO", "MONITOR", "GARAGEM", "QUARTOS", "CANETAS", "ACENDER", "APAGADA", 
    "DESLIGO", "DESERTO", "PLANETA", "GRAMADO", "LARANJA", "QUEIJOS", "TOMATES", "BANANAS", "PEPINOS", "FEIJÕES", 
    "BOLACHA", "CENOURA", "ABÓBORA", "GERENTE", "DIRETOR", "COMPRAS", "ESTOQUE", "BALANÇO", "DESPESA", "RECEITA", 
    "FATURAR", "TÍTULOS", "ARQUIVO", "IMAGENS", "FIGURAS", "LEGENDA", "CAPITAL", "CIDADES", "ADULTOS", "SENHORA",
    
    # 8 letras
    "TRABALHO", "PROBLEMA", "QUALQUER", "CONTROLE", "PROGRAMA", "TELEFONE", "INTERNET", "APRENDER", "MENSAGEM", 
    "RESPOSTA", "PERGUNTA", "MATERIAL", "PROCESSO", "NEGÓCIOS", "EMPRESAS", "MERCADOS", "SISTEMAS", "PROJETOS", 
    "CLIENTES", "PESSOAIS", "FAMÍLIAS", "CRIANÇAS", "FERIADOS", "SEGUNDOS", "MOMENTOS", "CAMINHOS", "DETALHES", 
    "AMBIENTE", "SOLUÇÕES", "PRÁTICAS", "ENTREGAS", "ESTOQUES", "DESPESAS", "RECEITAS", "BALANÇOS", "ARQUIVOS", 
    "LEGENDAS", "CADASTRO", "SUPORTES", "ELEMENTO", "CONCEITO", "CONTEXTO", "CONTEÚDO", "CAPÍTULO", "CAPITAIS", 
    "DISTRITO", "AVENIDAS", "RODOVIAS", "VIADUTOS", "PEQUENOS", "PEQUENAS", "ALTITUDE", "LARGURAS", "PROFUNDO", 
    "PROFUNDA", "ANTERIOR", "RECENTES", "PASSADOS", "SEGUINTE", "PRIMEIRO", "TERCEIRO", "VITÓRIAS", "DERROTAS", 
    "CORRENDO", "CHEGANDO", "VOLTANDO", "FECHANDO", "PASSANDO", "BUSCANDO", "CHAMANDO", "APRENDIA", "ESTUDAVA", 
    "PESQUISA", "ANALISTA", "MÉTODOS", "TEÓRICOS", "GESTORES", "GERENTES", "DIRETORA", "FORNECER", "CONSUMIR", 
    "COMPRAVA", "VENDERAM", "FECHARAM", "CORRERAM", "LEITORAS", "LEITORES", "UNIDADES", "CENTENAS", "MILHARES", 
    "MILHÕES", "BILHÕES", "SETORIAL", "REGIONAL", "NACIONAL", "ESTADUAL", "LIMPEZAS", "COZINHAS", "BANHEIRO", 
    "CADEIRAS",
    
    # 9 letras
    "APLICAÇÃO", "PROGRAMAS", "PROGRAMAR", "PROGRAMOU", "CONTROLES", "TELEFONES", "MENSAGENS", "RESPOSTAS", 
    "PERGUNTAS", "MATERIAIS", "PROCESSOS", "NEGOCIADO", "TRABALHAR", "TRABALHOS", "PROBLEMAS", "QUALIDADE", 
    "TREINANDO", "TREINADOR", "TREINADOS", "ATIVIDADE", "PLANEJADO", "PLANEJADA", "OBJETIVOS", "OBJETIVAS", 
    "OBJETIVAR", "OBJETIVOU", "APRENDIAM", "APRENDIDO", "DOCUMENTO", "ASSINANDO", "ASSINARAM", "ASSINANTE", 
    "COZINHADO", "BANHEIROS", "ECONOMIAS", "ECONÔMICO", "ECONÔMICA", "FATURADOS", "FATURADAS", "RECEBIDOS", 
    "RECEBIDAS", "DESPENSAS", "BALANCETE", "ESTOQUEAR", "COMPRADOR", "COMPRAMOS", "COMPRANDO", "COMPRARAM", 
    "VENDEDORA", "VENDERIAM", "ENTREGUEI", "ENTREGUES", "ENTREGARÁ", "PAGAMENTO", "COBRANÇAS", "CADASTRAR", 
    "CADASTROS", "REGISTROS", "CONSULTAR", "CONSULTAS", "RELATÓRIO", "PLANILHAS", "ORÇAMENTO", "COTAÇÕES", 
    "REPASSADO", "REPAGINAR", "ANALISADO", "ANALISADA", "AVALIADAS", "AVALIADOS", "AVALIANDO", "MELHORIAS", 
    "MELHORADO", "MELHORADA", "OTIMIZADO", "OTIMIZADA", "ORGANIZAR", "ORGANIZEI", "ORGANIZAM", "ORGANIZOU", 
    "PRIORIZAR", "PRIORIZEI", "PRIORIZAM", "PRIORIZOU", "SEPARANDO", "SEPARAMOS", "SEPARADAS", "SEPARADOS", 
    "CONECTADA", "CONECTADO", "DESLIGADO", "DESLIGADA", "DESLIGUEI", "LIGAMENTO", "ATUALIZAR", "ATUALIZEI", 
    "ATUALIZAM", "ATUALIZOU", "APROVAÇÃO", "RESOLVIDO",
    
    # 10 letras
    "APLICAÇÕES", "INFORMAÇÃO", "QUALIDADES", "PLANEJAMOS", "APROVEITAR", "APROVEITAM", "APROVEITOU", "APRIMORAR", 
    "APRENDEMOS", "APRENDENDO", "CONHECEMOS", "CONHECENDO", "COMBINAMOS", "COMBINANDO", "CONSIDERAR", "CONSIDERAM", 
    "CONSIDEROU", "CONTRATADA", "CONTRATADO", "CONSTRUÇÃO", "MELHORANDO", "MELHORAMOS", "MELHORARAM", "MELHORIAS", 
    "OTIMIZANDO", "OTIMIZAMOS", "OTIMIZADOR", "ORGANIZADA", "ORGANIZADO", "PRIORIDADE", "FINANCEIRO", "FINANCEIRA", 
    "FATURADORA", "RECEBERMOS", "RECEBERIAM", "PAGAMENTOS", "COBRADORES", "CADASTRADO", "CADASTRADA", "REGISTRADO", 
    "REGISTRADA", "RELATÓRIOS", "ORÇAMENTOS", "AVALIAÇÕES", "OTIMIZAÇÃO", "CONECTADAS", "CONECTADOS", "DESLIGADAS", 
    "DESLIGADOS", "LIGAMENTOS", "ATUALIZADO", "ATUALIZADA", "RESOLVIDOS", "RESOLVIDAS", "DOCUMENTOS", "DOCUMENTAR", 
    "DOCUMENTOU", "ASSINATURA", "COZINHEIRO", "COZINHEIRA", "ECONÔMICOS", "ECONÔMICAS", "ENTREGAMOS", "ENTREGANDO", 
    "ENTREGADOR", "ENTREGARAM", "ENTREGAREI", "CONSULTADO", "CONSULTADA", "CONSULTORA", "RELACIONAR", "RELACIONOU", 
    "RELACIONAM", "APRESENTAR", "APRESENTOU", "APRESENTAM", "DISPONÍVEL", "POSSÍVEIS", "CATEGORIAS", "CRONOGRAMA", 
    "PORTFÓLIOS", "BIBLIOTECA", "APRIMORADO", "APRIMORADA", "SEGMENTADO", "SEGMENTOS", "APLICATIVO", "COMPUTADOR", 
    "FERRAMENTA", "RESULTADOS", "DIFERENTES", "FACILMENTE", "LENTAMENTE", "CLARAMENTE", "CONFIGURAR", "GARANTIMOS", 
    "GARANTINDO", "GARANTIDOS", "GARANTIDAS", "QUALIFICAR"
}

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

    # ✅ Adiciona as palavras comuns de referência ao dicionário
    dicionario.update(PALAVRAS_COMUNS_REF)
    for p in PALAVRAS_COMUNS_REF:
        for i in range(1, len(p) + 1):
            prefixos.add(p[:i])

    return dicionario, prefixos


# =========================================================
# BOGGLE SOLVER (DFS)
# =========================================================
def buscar_palavras_boggle(matriz, dicionario, prefixos):
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
        nova = palavra + letra

        if nova not in prefixos:
            return

        # ✅ melhoria: mínimo 3 letras
        if nova in dicionario and len(nova) >= MIN_PALAVRA and nova not in achadas:
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
    # pega do Streamlit secrets (sem input)
    return (
        st.secrets.get("GOOGLE_API_KEY", "")
        or st.secrets.get("GEMINI_API_KEY", "")
    )

def extrair_matriz_google(imagem: Image.Image, linhas: int, colunas: int):
    api_key = _get_api_key()
    if not api_key:
        raise ValueError('A GOOGLE_API_KEY ou GEMINI_API_KEY não foi configurada nos Secrets do Streamlit.')

    genai.configure(api_key=api_key)

    modelo_id = _escolher_modelo_generate_content()
    model = genai.GenerativeModel(modelo_id)

    img_resized = imagem.copy()
    img_resized.thumbnail((1400, 1400))

    # 1) primeira tentativa
    resp1 = model.generate_content([_prompt_extracao(linhas, colunas), img_resized], request_options={"timeout": 120})
    j1 = extrair_json_estrito(resp1.text)
    m1 = sanear_matriz(j1.get("matriz"))

    try:
        m_ok = ajustar_dimensoes(m1, linhas, colunas)
        return modelo_id, m_ok, j1
    except Exception:
        pass

    # 2) retry: correção de dimensões
    resp2 = model.generate_content([_prompt_correcao(linhas, colunas, m1)], request_options={"timeout": 120})
    j2 = extrair_json_estrito(resp2.text)
    m2 = sanear_matriz(j2.get("matriz"))

    m_ok = ajustar_dimensoes(m2, linhas, colunas)
    return modelo_id, m_ok, j2


# =========================================================
# UI PRINCIPAL
# =========================================================
st.sidebar.header("⚙️ Configurações")

tamanho_selecionado = st.sidebar.selectbox(
    "Tamanho do tabuleiro:",
    options=list(TAMANHOS.keys()),
    index=0
)

linhas = colunas = TAMANHOS[tamanho_selecionado]

arquivo_imagem = st.file_uploader("📤 Upload da imagem do tabuleiro", type=["png", "jpg", "jpeg"])

if arquivo_imagem:
    imagem = Image.open(arquivo_imagem)
    st.image(imagem, caption="Tabuleiro enviado", use_container_width=True)

    with st.spinner("🔄 Extraindo matriz da imagem via Gemini..."):
        try:
            modelo_id, matriz, json_completo = extrair_matriz_google(imagem, linhas, colunas)
            
            st.success(f"✅ Matriz extraída com sucesso! (Modelo: `{modelo_id}`)")
            
            with st.expander("🔍 Ver matriz extraída"):
                df_matriz = pd.DataFrame(matriz)
                st.dataframe(df_matriz, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erro ao extrair a matriz: {e}")
            st.stop()

    with st.spinner("🔍 Buscando palavras no tabuleiro..."):
        try:
            dicionario, prefixos = carregar_dicionario_pt()
            achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)
            palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))

            st.success(f"✅ {len(palavras)} palavras encontradas!")

            # ✅ PALAVRA-CHAVE AUTOMÁTICA
            n_tabuleiro = len(matriz)
            palavra_chave = escolher_palavra_chave(palavras, n_tabuleiro)

            st.subheader("🎯 Palavra-chave sugerida")
            if palavra_chave:
                st.markdown(f"**{palavra_chave}**  \n_Tamanho: {len(palavra_chave)} letras_")
            else:
                st.write("Não foi possível sugerir palavra-chave nesta rodada.")

            st.divider()

            # Lista de palavras
            st.subheader("📝 Palavras encontradas")
            
            df_palavras = pd.DataFrame({
                "Palavra": palavras,
                "Tamanho": [len(p) for p in palavras]
            })
            
            st.dataframe(df_palavras, use_container_width=True, height=400)

            # Download
            csv = df_palavras.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar lista de palavras (CSV)",
                data=csv,
                file_name=f"palavras_{tamanho_selecionado}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Erro ao buscar palavras: {e}")
            st.code(traceback.format_exc())

else:
    st.info("👆 Faça upload de uma imagem do tabuleiro para começar.")
