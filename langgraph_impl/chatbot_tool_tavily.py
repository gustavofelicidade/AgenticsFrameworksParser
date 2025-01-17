"""
chatbot_tool_tavily.py
=======================

Este módulo demonstra um exemplo de implementação de chatbot com LangGraph,
fazendo uso da ferramenta TavilySearchResults para pesquisa na web. Nesta
versão, trazemos a "Parte 2 (metade final)" do tutorial de LangGraph,
adaptada para rodar em um script .py, com comentários em Português,
explicando a lógica de roteamento condicional e a introdução do uso de
ferramentas via `BasicToolNode`.

As principais novidades em relação à Parte 1 são:
1. Uso do `llm_with_tools` para que o LLM tenha conhecimento do formato
   correto de JSON para chamadas de ferramentas (tool calls).
2. Implementação de um nó de ferramentas (`BasicToolNode`) que executa
   chamadas de ferramentas caso o modelo de linguagem solicite.
3. Criação de bordas condicionais (`add_conditional_edges`) que redirecionam
   o fluxo do grafo para as ferramentas, caso seja solicitado, ou para o
   ponto de término (END), caso não seja necessária a ferramenta.

Requisitos mínimos:
- langchain_anthropic
- langgraph
- langchain_community
- tqdm, matplotlib, rich, pprint (para fins de demonstração e vizualização)
- (Opcional) langchain_core.messages para ToolMessage
"""

import time
import json
import matplotlib.pyplot as plt
from typing import Annotated
from typing_extensions import TypedDict
from pprint import pprint
from tqdm import tqdm

# Pacotes para exibição colorida e legível de logs
from rich import print
from rich.pretty import Pretty

# Importações específicas do LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LLM Anthropic + Ferramenta Tavily
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

# Import do ToolMessage para encapsular as respostas das ferramentas
# (Disponível em algumas instalações do LangChain; pode variar)
try:
    from langchain_core.messages import ToolMessage
except ImportError:
    # Se não estiver disponível, teremos apenas um fallback aqui.
    class ToolMessage:
        def __init__(self, content, name, tool_call_id):
            self.content = content
            self.name = name
            self.tool_call_id = tool_call_id

# -----------------------------------------------------------------------------
# 1) Configuração da ferramenta Tavily Search
# -----------------------------------------------------------------------------
# Esta ferramenta permite ao LLM realizar pesquisas na web por meio do Tavily.
# Aqui, max_results=2 limita o número de resultados retornados.

tool = TavilySearchResults(max_results=2)
tools = [tool]

def configure_tools():
    """
    Função auxiliar para (re)configurar as ferramentas,
    caso quiséssemos mudar parâmetros dinâmicos.
    """
    tool = TavilySearchResults(max_results=2)
    return [tool]

# -----------------------------------------------------------------------------
# 2) Definição do estado (State) do Chatbot para o LangGraph
# -----------------------------------------------------------------------------
# Neste caso, nosso estado carrega apenas a lista de mensagens trocadas.

class State(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------------------------------------------------------
# 3) Criação do "graph_builder" e configuração do LLM
# -----------------------------------------------------------------------------
# O LangGraph fornece a classe StateGraph para construirmos passo a passo
# a lógica de nós (nodes) e conexões (edges) de nosso fluxo de conversa.

graph_builder = StateGraph(State)

# Configurando o modelo LLM Anthropic.
# O "bind_tools" permite que este LLM saiba chamar as ferramentas definidas.
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------------------------
# 4) Nó "chatbot": interage com o LLM
# -----------------------------------------------------------------------------
# Este nó recebe o estado e retorna a nova mensagem do LLM. Observe que
# agora invocamos `llm_with_tools`, que permite ao LLM fazer chamadas
# de ferramentas (tool calls).

def chatbot(state: State):
    """
    Nó de processamento principal do chatbot, que interage com o LLM.
    Se o LLM solicitar uso de ferramenta, a mensagem retornada virá com
    as instruções de tool_calls para que outro nó (tools) possa executar.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# -----------------------------------------------------------------------------
# 5) Nó "BasicToolNode": executa chamadas de ferramentas, se houver
# -----------------------------------------------------------------------------
# Esta classe implementa a lógica de ver se a última mensagem do LLM
# continha chamadas de ferramentas. Se sim, invoca a ferramenta e retorna
# a resposta como ToolMessage.

class BasicToolNode:
    """Um nó que executa as ferramentas solicitadas na última mensagem do LLM."""

    def __init__(self, tools: list) -> None:
        # Indexamos as ferramentas por nome, para acessar rapidamente
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """
        A função principal do nó.
        Verifica se a última mensagem tem 'tool_calls' e invoca as ferramentas.
        """
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("Nenhuma mensagem encontrada no input para BasicToolNode.")

        outputs = []
        # Se houver chamadas de ferramenta, iteramos e invocamos cada uma
        for tool_call in getattr(message, "tool_calls", []):
            # Invocando a ferramenta pelo nome
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        # Retornamos as novas mensagens (ToolMessages) resultantes das ferramentas
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# -----------------------------------------------------------------------------
# 6) Definição do roteamento condicional (conditional_edges)
# -----------------------------------------------------------------------------
# Precisamos dizer ao LangGraph como rotear o fluxo após o nó "chatbot".
# Se o LLM solicitou uso de alguma ferramenta (tool_calls), enviamos o
# controle para o nó "tools". Caso contrário, encerramos (END).
# Depois que executamos o nó "tools", retornamos ao "chatbot" para processar
# a resposta da ferramenta.

def route_tools(state: State):
    """
    Função de roteamento que verifica se a última mensagem gerada pelo chatbot
    contém tool calls. Se contiver, retorna "tools", senão retorna END
    para encerrar o fluxo.
    """
    if isinstance(state, list):
        # Caso o State seja apenas uma lista de mensagens
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        # Caso o State seja um dicionário com chave "messages"
        ai_message = messages[-1]
    else:
        raise ValueError(f"Nenhuma mensagem encontrada no estado para route_tools: {state}")

    # Se a última mensagem tiver tool_calls, redirecionamos para o nó "tools"
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    # Caso contrário, encerramos o fluxo
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# Quando a ferramenta terminar de executar, voltamos ao chatbot
graph_builder.add_edge("tools", "chatbot")

# Indica que o fluxo começa no nó "chatbot"
graph_builder.add_edge(START, "chatbot")

# -----------------------------------------------------------------------------
# 7) Compilar o grafo
# -----------------------------------------------------------------------------
graph = graph_builder.compile()

# -----------------------------------------------------------------------------
# 8) Função de visualização do gráfico em ASCII
# -----------------------------------------------------------------------------
def visualize_graph_ascii(graph):
    """
    Tenta renderizar o gráfico em formato ASCII e imprimir no console.
    Pode falhar caso não haja suporte local.
    """
    try:
        ascii_graph = graph.get_graph().draw_ascii()
        print("\n[bold blue]Gráfico do Chatbot (Formato ASCII):[/bold blue]")
        print(ascii_graph)
    except Exception as e:
        print("Erro ao renderizar o gráfico em ASCII:", e)

# -----------------------------------------------------------------------------
# 9) Função de fluxo de mensagens do chatbot
# -----------------------------------------------------------------------------
def stream_graph_updates(user_input: str):
    """
    Roda o grafo para processar a entrada do usuário, exibindo as respostas
    à medida que são geradas. Ele utiliza o método stream do grafo, que
    emite eventos a cada nó executado.
    """
    # Passamos a mensagem do usuário como ("user", user_input) no 'messages'
    for event in graph.stream({"messages": [("user", user_input)]}):
        # Cada evento contém o estado atualizado do grafo
        for value in event.values():
            # Acessar diretamente o conteúdo da última mensagem
            response = value["messages"][-1].content
            print("\n[bold green]Assistant:[/bold green]", response)

# -----------------------------------------------------------------------------
# 10) Função auxiliar para usar a ferramenta Tavily Search diretamente
# -----------------------------------------------------------------------------
def use_tool(user_query: str):
    """
    Exemplo de função que chama explicitamente a ferramenta TavilySearchResults,
    sem passar pelo fluxo principal do grafo.
    """
    tool = tools[0]
    results = tool.invoke(user_query)
    print("\n[bold yellow]Resultados da busca:[/bold yellow]")
    pprint(results)

# -----------------------------------------------------------------------------
# 11) Execução principal (main)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Visualizar o grafo no console (em formato ASCII)
    visualize_graph_ascii(graph)

    # Loop de interação com o usuário via terminal
    print("\n[bold cyan]Chatbot iniciado! Digite sua mensagem.[/bold cyan]")
    print("[yellow]Digite 'quit', 'exit' ou 'q' para sair.[/yellow]\n")

    while True:
        try:
            user_input = input("[bold blue]User:[/bold blue] ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("[bold magenta]Até mais![/bold magenta]")
                break

            # Se a entrada começar com "search:", chamamos a ferramenta direto
            if user_input.lower().startswith("search:"):
                print("Usando a ferramenta 'search:' ")
                query = user_input.split("search:", 1)[1].strip()
                use_tool(query)
            else:
                # Caso contrário, processamos pelo grafo
                stream_graph_updates(user_input)

        except KeyboardInterrupt:
            print("\n[bold red]Chat encerrado pelo usuário.[/bold red]")
            break
        except Exception as e:
            print(f"\n[bold red]Erro inesperado:[/bold red] {e}")
