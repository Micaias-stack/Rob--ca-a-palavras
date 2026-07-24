import io
import json
import re
import time
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
st.set_page_config(page_title="Robô Caça-Palavras (Boggle)", page_icon=None, layout="wide")
st.title("Robô Caça-Palavras (Gemini + Python)")
st.caption("Upload da foto → Gemini extrai a grade → Python encontra palavras (PT-BR).")


# =========================================================
# REGRAS / MELHORIAS
# =========================================================
TAMANHOS = {"4x4": 4, "5x5": 5, "6x6": 6}
MIN_PALAVRA = 3


# =========================================================
# LISTA DE PALAVRAS COMUNS (REFERÊNCIA DO USUÁRIO)
# - Usos:
#   1) (opcional) reforçar dicionário: garantir que essas palavras sejam aceitas
#   2) palavra-chave: evitar cair em termos "óbvios demais" quando possível
# =========================================================
PALAVRAS_COMUNS_REF_TEXT = r"""
100 palavras comuns de três letras (sem nomes próprios):
que, com, por, uma, nas, nos, das, dos, sem, aos, mas, pra, pro, tem, foi, sou, era, ser, ver, seu, sua, meu, teu, tua, ele
ela, nós, não, sim, nem, dez, mil, cem, vez, ano, dia, bem, mal, bom, boa, dor, cor, luz, sol, paz, fez, vai, vem, vou, usa
uso, sob, lhe, tal, num, uns, mim, ali, ora, oco, tão, mão, pão, são, dar, ler, rir, pôr, tia, tio, pai, mãe, avó, avô, faz
diz, lei, lua, mar, via, rua, voz, som, sai, viu, deu, cai, pôs, põe, ato, eis, cal, mel, sal, chá, céu, olá, eco, gás, sul

4 letras: para, mais, como, onde, aqui, hoje, caso, pois, pela, pelo, este, esta, isto, isso, dele, dela, nele, nela, cada, algo, nada, tudo, vida, casa, lado
modo, hora, dias, anos, bens, ruas, ruim, alto, leve, doce, pode, quer, fala, fica, veja, você, eles, elas, dois, três, sete, oito, nove, doze, logo
cedo, sair, saiu, veio, come, comi, bebe, bebi, leva, pago, paga, abre, abri, amei, mora, moro, cala, cale, riem, riso, veem, pôde, após, azul, rosa
roxo, bege, flor, frio, neve, maré, lago, mata, mesa, cama, sofá, saco, vela, fogo, tela, pano, pote, tubo, peça, leis, reis, mães, cada, nada, tudo

5 letras: muito, assim, ainda, pouco, tempo, gente, coisa, parte, vezes, nunca, sobre, fazer, poder, dever, deixa, dessa, desse, deste, mesmo, menos, agora, antes, noite, tarde, certo, seria, saber, achar, ficar, levar, pedir, dizer, falar, viver, mundo, livro, jovem, idoso, homem, carro, casas, ruins, altos, baixo, leves, doces, claro, preto, verde, azuis, cinza, feliz, chato, cheio, vazio, forte, fraco, curto, longo, outro, mesma, todos, todas, maior, menor, demais, acima, entre, desde, ontem, ceder, saída, chega, feita, feito, disse, fomos, foram, estar, estou, estão, serão, teria, houve, haver, posso, possa, sabem, quero, vamos, vimos, busca, achou, trago, leito, lemas, temas, itens, dados, fatos.

6letras: porque, depois, embora, apenas, sempre, jamais, dentro, abaixo, frente, apesar, pessoa, escola, amigos, amigas, comida, bebida, salada, cidade, estado, bairro, centro, planta, animal, imagem, tarefa, estudo, matriz, planos, tomada, seguro, mínimo, máximo, rápido, modelo, método, teoria, código, tópico, página, título, listas, textos, frases, acesso, senhas, moedas, preços, custos, vendas, compra, boleto, cartão, débito, equipe, corpos, cabeça, braços, pernas, lábios, cabelo, costas, ouvido, vacina, doença, doente, febres, câncer, medida, quilos, litros, metros, gramas, grande, pronto, pronta, quente, escuro, gelado, alegre, triste, justos, certos, errado, errada, branco, cinzas, amanhã, quarta, quinta, sábado, terças, correr, vender, buscar, chegar, voltar, fechar, passar, objeto, motivo.

7letras: exemplo, pessoas, família, criança, preciso, precisa, caminho, momento, segundo, mercado, sistema, produto, serviço, cuidado, celular, projeto, reunião, viagens, domingo, segunda, limpeza, cozinha, cadeira, tijolos, amarelo, alegria, objetos, garrafa, sapatos, camisas, armário, caderno, leitura, sorriso, pintura, retrato, desenho, cliente, empresa, negócio, estados, centros, estrada, avenida, rodovia, viaduto, pequeno, pequena, maioria, meninos, meninas, abrindo, pagando, aplicar, aceitar, ajudado, cansado, cansada, felizes, tristes, certeza, dúvidas, matéria, teclado, monitor, garagem, quartos, canetas, acender, apagada, desligo, deserto, planeta, gramado, laranja, queijos, tomates, bananas, pepinos, feijões, bolacha, cenoura, abóbora, gerente, diretor, compras, estoque, balanço, despesa, receita, faturar, títulos, arquivo, imagens, figuras, legenda, capital, cidades, adultos, senhora.

8 letras: trabalho, problema, qualquer, controle, programa, telefone, internet, aprender, mensagem, resposta, pergunta, material, processo, negócios, empresas, mercados, sistemas, projetos, clientes, pessoais, famílias, crianças, feriados, segundos, momentos
caminhos, detalhes, ambiente, soluções, práticas, entregas, estoques, despesas, receitas, balanços, arquivos, legendas, cadastro, suportes, elemento, conceito, contexto, conteúdo, capítulo, capitais, distrito, avenidas, rodovias, viadutos, pequenos
pequenas, altitude, larguras, profundo, profunda, anterior, recentes, passados, seguinte, primeiro, terceiro, vitórias, derrotas, correndo, chegando, voltando, fechando, passando, buscando, chamando, aprendia, estudava, pesquisa, analista, métodos
teóricos, gestores, gerentes, diretora, fornecer, consumir, comprava, venderam, fecharam, correram, leitoras, leitores, unidades, centenas, milhares, milhões, bilhões, setorial, regional, nacional, estadual, limpezas, cozinhas, banheiro, cadeiras

9 letras: aplicação, programas, programar, programou, controles, telefones, mensagens, respostas, perguntas, materiais, processos, negociado, trabalhar, trabalhos, problemas, qualidade, treinando, treinador, treinados, atividade, planejado, planejada, objetivos, objetivas, objetivar
objetivou, aprendiam, aprendido, documento, assinando, assinaram, assinante, cozinhado, banheiros, economias, econômico, econômica, faturados, faturadas, recebidos, recebidas, despensas, balancete, estoquear, comprador, compramos, comprando, compraram, vendedora, venderiam
entreguei, entregues, entregará, pagamento, cobranças, cadastrar, cadastros, registros, consultar, consultas, relatório, planilhas, orçamento, cotações, repassado, repaginar, analisado, analisada, avaliadas, avaliados, avaliando, melhorias, melhorado, melhorada, otimizado
otimizada, organizar, organizei, organizam, organizou, priorizar, priorizei, priorizam, priorizou, separando, separamos, separadas, separados, conectada, conectado, desligado, desligada, desliguei, ligamento, atualizar, atualizei, atualizam, atualizou, aprovação, resolvido

10 letras: aplicações, informação, qualidades, planejamos, aproveitar, aproveitam, aproveitou, aprimorar, aprendemos, aprendendo, conhecemos, conhecendo, combinamos, combinando, considerar, consideram, considerou, contratada, contratado, construção, melhorando, melhoramos, melhoraram, melhorias, otimizando
otimizamos, otimizador, organizada, organizado, prioridade, financeiro, financeira, faturadora, recebermos, receberiam, pagamentos, cobradores, cadastrado, cadastrada, registrado, registrada, relatórios, orçamentos, avaliações, otimização, conectadas, conectados, desligadas, desligados, ligamentos
atualizado, atualizada, resolvidos, resolvidas, documentos, documentar, documentou, assinatura, cozinheiro, cozinheira, econômicos, econômicas, entregamos, entregando, entregador, entregaram, entregarei, consultado, consultada, consultora, relacionar, relacionou, relacionam, apresentar, apresentou
apresentam, disponível, possíveis, categorias, cronograma, portfólios, biblioteca, aprimorado, aprimorada, segmentado, segmentos, aplicativo, computador, ferramenta, resultados, diferentes, facilmente, lentamente, claramente, configurar, garantimos, garantindo, garantidos, garantidas, qualificar
"""

def _normalizar_ref_words(texto: str) -> set[str]:
    # pega só tokens com letras (inclui acentos), ignora números/pontuação
    toks = re.findall(r"[A-Za-zÀ-ÿ]+", texto, flags=re.UNICODE)
    out = set()
    for t in toks:
        tt = t.strip().upper()
        if len(tt) >= 3:
            out.add(tt)
    return out

PALAVRAS_COMUNS_REF = _normalizar_ref_words(PALAVRAS_COMUNS_REF_TEXT)


# =========================================================
# PALAVRA-CHAVE
# =========================================================
def _min_len_chave_por_tabuleiro(n: int) -> int:
    return {4: 6, 5: 7, 6: 8}.get(n, 7)

def escolher_palavra_chave(palavras_encontradas: list[str], n_tabuleiro: int) -> str:
    """
    Prioridade:
    1) palavra longa (>= mínimo por tabuleiro; senão, as mais longas)
    2) evitar cair em palavra "óbvia" (lista PALAVRAS_COMUNS_REF), se houver alternativa
    3) se wordfreq estiver disponível, preferir “cotidiano mas não muito comum”
       (Zipf ~ 3.0 a 4.6, alvo 3.8), mantendo comprimento como peso principal
    """
    if not palavras_encontradas:
        return ""

    min_len = _min_len_chave_por_tabuleiro(n_tabuleiro)
    cand = [w for w in palavras_encontradas if len(w) >= min_len]

    if not cand:
        maxlen = max(len(w) for w in palavras_encontradas)
        cand = [w for w in palavras_encontradas if len(w) == maxlen]

    # tenta evitar as super comuns, se der
    cand_n_comum = [w for w in cand if w not in PALAVRAS_COMUNS_REF]
    if cand_n_comum:
        cand = cand_n_comum

    if WORDFREQ_OK:
        def z(w: str) -> float:
            return float(zipf_frequency(w.lower(), "pt"))

        # filtra extremos: nem comum demais, nem obscura demais
        cand_mid = [w for w in cand if 3.0 <= z(w) <= 4.6]
        if cand_mid:
            cand = cand_mid

        alvo = 3.8
        def score(w: str):
            # comprimento manda; frequência só desempata/ajuda
            return (len(w), -abs(z(w) - alvo), 0 if w not in PALAVRAS_COMUNS_REF else -1)

        return max(cand, key=score)

    cand.sort(key=lambda w: (-len(w), w))
    return cand[0]


# =========================================================
# DICIONÁRIO PT-BR (CACHE)
# =========================================================
@st.cache_data(ttl=21600)  # 6h
def carregar_dicionario_pt():
    url = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
    with urllib.request.urlopen(url, timeout=60) as resposta:
        palavras_raw = resposta.read().decode("utf-8").splitlines()

    dicionario = set()
    prefixos = set()

    def add_word(p: str):
        p = (p or "").upper().strip()
        if len(p) >= 2 and "-" not in p and "." not in p:
            dicionario.add(p)
            for i in range(1, len(p) + 1):
                prefixos.add(p[:i])

    # base
    for p in palavras_raw:
        add_word(p)

    # reforço (palavras aceitas pelo seu jogo / referência)
    for p in PALAVRAS_COMUNS_REF:
        add_word(p)

    return dicionario, prefixos


# =========================================================
# BOGGLE SOLVER (DFS) - mínimo 3 letras
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
        if not letra:
            return

        nova = palavra + letra

        if nova not in prefixos:
            return

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

    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"JSON não encontrado na resposta do modelo.\nResposta (início): {t[:400]}")
    return json.loads(m.group(0))

def _cell_ok(x: str) -> bool:
    s = (x or "").strip().upper()
    return bool(re.search(r"[A-ZÀ-Ü]", s))

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
        st.warning(f"O retorno veio com {len(matriz)} linhas; vou considerar apenas as primeiras {linhas}.")
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

def parse_matriz_manual(texto: str):
    """
    Aceita:
    1) JSON: {"matriz":[["A","B"],...]}
    2) Linhas: uma linha por linha da grade, separado por espaço/vírgula.
    """
    t = (texto or "").strip()
    if not t:
        return None

    try:
        j = extrair_json_estrito(t)
        m = j.get("matriz")
        return sanear_matriz(m)
    except Exception:
        pass

    linhas = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\s;]+", line) if p]
        linhas.append(parts)

    return sanear_matriz(linhas)


# =========================================================
# GEMINI: EXTRAÇÃO DA MATRIZ (com retry/429 + fallback de modelo)
# =========================================================
def _get_api_key():
    return st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")

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
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("Chave não configurada. Defina GOOGLE_API_KEY (ou GEMINI_API_KEY) nos Secrets.")

    genai.configure(api_key=api_key)

    modelos_preferidos = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    img_resized = imagem.copy().convert("RGB")
    img_resized.thumbnail((1400, 1400))

    last_err = None

    for modelo_id in modelos_preferidos:
        try:
            model = genai.GenerativeModel(modelo_id)

            # Tentativa 1: extrair
            for tent in range(3):
                try:
                    resp1 = model.generate_content(
                        [_prompt_extracao(linhas, colunas), img_resized],
                        request_options={"timeout": 120},
                    )
                    j1 = extrair_json_estrito(resp1.text)
                    m1 = sanear_matriz(j1.get("matriz"))

                    # se já veio perfeito
                    try:
                        m_ok = ajustar_dimensoes(m1, linhas, colunas)
                        return modelo_id, m_ok, j1
                    except Exception:
                        pass

                    # Tentativa 2: corrigir
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
            # falhou (inclusive 429): tenta próximo modelo
            continue

    raise RuntimeError(f"Falhou em todos os modelos. Último erro: {last_err}")

@st.cache_data(ttl=86400, show_spinner=False)  # 24h
def extrair_matriz_google_cached(image_bytes: bytes, linhas: int, colunas: int):
    imagem = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return extrair_matriz_google(imagem, linhas, colunas)


# =========================================================
# UI
# =========================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    tamanho_escolhido = st.selectbox("Tamanho do tabuleiro", list(TAMANHOS.keys()), index=0)
    n = TAMANHOS[tamanho_escolhido]
    linhas = n
    colunas = n

    arquivo = st.file_uploader("Envie a foto do tabuleiro", type=["png", "jpg", "jpeg", "webp"])

    st.caption("Fallback manual (se a API estiver sem cota): cole a matriz abaixo.")
    exemplo = (
        "A B C D\n"
        "E F G H\n"
        "I J K L\n"
        "M N O P"
    ) if n == 4 else (
        "A B C D E\n"
        "F G H I J\n"
        "K L M N O\n"
        "P Q R S T\n"
        "U V W X Y"
    )
    matriz_manual_text = st.text_area(
        f"Matriz manual ({n}x{n})",
        value="",
        height=170,
        placeholder=exemplo,
    )

with col2:
    matriz = None
    modelo_usado = None
    erro_extracao = None

    if arquivo is not None:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="Imagem enviada", use_container_width=True)

        api_key = _get_api_key()
        if api_key:
            with st.spinner("Extraindo matriz com Gemini..."):
                try:
                    image_bytes = _img_to_bytes(imagem)
                    modelo_usado, matriz, _raw = extrair_matriz_google_cached(image_bytes, linhas, colunas)
                except Exception as e:
                    erro_extracao = e
        else:
            erro_extracao = ValueError("Sem API key configurada.")

    # Se não tiver matriz pela API, tenta manual
    if matriz is None and matriz_manual_text.strip():
        try:
            m = parse_matriz_manual(matriz_manual_text)
            if m is not None:
                matriz = ajustar_dimensoes(m, linhas, colunas)
        except Exception as e:
            st.error(f"Matriz manual inválida: {e}")

    if matriz is None:
        if erro_extracao is not None:
            if _is_429(erro_extracao):
                st.warning("A API retornou 429 (quota exceeded). Use a matriz manual para continuar.")
            else:
                st.error(f"Não consegui extrair a matriz: {erro_extracao}")
        st.stop()

    if modelo_usado:
        st.caption(f"Modelo: {modelo_usado}")

    st.subheader("Matriz final")
    st.dataframe(pd.DataFrame(matriz), use_container_width=True, hide_index=True)


# =========================================================
# SOLVER
# =========================================================
with st.spinner("Carregando dicionário PT-BR..."):
    dicionario, prefixos = carregar_dicionario_pt()

with st.spinner("Buscando palavras..."):
    achadas = buscar_palavras_boggle(matriz, dicionario, prefixos)

palavras = sorted(achadas.keys(), key=lambda w: (-len(w), w))
palavra_chave = escolher_palavra_chave(palavras, n_tabuleiro=n)

st.subheader("Palavra-chave sugerida")
if palavra_chave:
    st.write(f"**{palavra_chave}**")
    st.caption("Critérios: longa; evita muito comum quando possível; e (se disponível) frequência 'cotidiano mas não comum'.")
else:
    st.write("Não foi possível sugerir palavra-chave.")

st.subheader("Palavras encontradas")
st.write(f"Total: **{len(palavras)}** (mínimo {MIN_PALAVRA} letras)")

# Lista simples (mais leve que mostrar caminho)
df = pd.DataFrame(
    {"palavra": palavras, "tamanho": [len(w) for w in palavras], "comum_ref": [w in PALAVRAS_COMUNS_REF for w in palavras]}
).sort_values(["tamanho", "palavra"], ascending=[False, True])

st.dataframe(df, use_container_width=True, hide_index=True)
