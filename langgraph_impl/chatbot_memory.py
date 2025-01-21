"""
chatbot_memory.py
=================

Este módulo demonstra como adicionar memória (checkpointing) ao chatbot
construído com LangGraph, usando a ferramenta TavilySearchResults e o
modelo ChatAnthropic. Agora, cada interação do usuário é salva no estado,
de forma que, ao reutilizar o mesmo thread_id, o contexto do histórico
de conversas é recuperado e permite um bate-papo multi-turn coerente.

Este script se baseia no 'chatbot_tool_tavily.py' da Parte 2 do tutorial,
mas substitui o BasicToolNode e o roteamento manual por ferramentas
pelas implementações pré-construídas (ToolNode e tools_condition) do
LangGraph, além de incorporar um checkpointer MemorySaver para persistir
o estado em memória.
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

# --------------------------------------------------------------------------
# Importações específicas do LangGraph e da ferramenta Tavily
# --------------------------------------------------------------------------
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Novo: Import do MemorySaver para manter o histórico (checkpointing)
from langgraph.checkpoint.memory import MemorySaver

# Classes e métodos do LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Nós e condição pré-construídos que substituem a lógica manual de tools
from langgraph.prebuilt import ToolNode, tools_condition

# -----------------------------------------------------------------------------
# 1) Definição de State e criação do "graph_builder"
# -----------------------------------------------------------------------------
# Nosso estado (State) novamente carrega apenas a lista de mensagens trocadas,
# mas agora será salvo a cada passo por meio do MemorySaver.

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# -----------------------------------------------------------------------------
# 2) Configuração do checkpointer (memória)
# -----------------------------------------------------------------------------
# Usamos MemorySaver para manter o histórico em memória. Em produção, você
# poderia usar outro tipo de checkpointer (SqliteSaver, PostgresSaver, etc.).

memory = MemorySaver()

# -----------------------------------------------------------------------------
# 3) Configuração do LLM e ferramenta
# -----------------------------------------------------------------------------
# A ferramenta de busca TavilySearchResults (limitada a 2 resultados).
tool = TavilySearchResults(max_results=2)
tools = [tool]

# O modelo ChatAnthropic que iremos utilizar,
# vinculado às ferramentas (bind_tools) para permitir tool calls.
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------------------------
# 4) Nó chatbot: interage com o LLM
# -----------------------------------------------------------------------------
def chatbot(state: State):
    """
    Nó de processamento principal do chatbot, que delega ao llm_with_tools.
    O output pode ter ou não chamadas de ferramenta (tool_calls).
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# -----------------------------------------------------------------------------
# 5) Nó de ferramentas ToolNode (pré-construído)
# -----------------------------------------------------------------------------
# Substitui o BasicToolNode da Parte 2. O ToolNode já sabe executar
# as ferramentas, caso o modelo tenha gerado chamadas.
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# -----------------------------------------------------------------------------
# 6) Roteamento condicional com tools_condition (pré-construído)
# -----------------------------------------------------------------------------
# O tools_condition verifica se houve tool_calls. Se sim, vai para "tools".
# Caso contrário, vai para END. Esse comportamento é idêntico à lógica manual
# que escrevemos anteriormente, porém encapsulado em uma função pronta.

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Quando as ferramentas terminam de rodar, voltamos ao chatbot
graph_builder.add_edge("tools", "chatbot")

# O fluxo começa no "chatbot"
graph_builder.add_edge(START, "chatbot")

# -----------------------------------------------------------------------------
# 7) Compilar o grafo com MemorySaver
# -----------------------------------------------------------------------------
# Ao compilar, fornecemos o checkpointer=memory. Cada execução do grafo
# salvando e/ou carregando o estado com base em um thread_id.
graph = graph_builder.compile(checkpointer=memory)

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
# 9) Fluxo de mensagens do chatbot
# -----------------------------------------------------------------------------
def stream_graph_updates(user_input: str, thread_id: str = "1"):
    """
    Executa o grafo usando thread_id para persistir a memória.
    Cada nó do grafo é executado em sequência; se o chatbot chamar uma ferramenta,
    iremos ao 'tools', depois voltamos ao 'chatbot', etc.
    """
    # Ao chamar graph.stream, passamos config={"configurable": {"thread_id": ...}}
    # para o checkpointer salvar/carregar o estado dessa conversa.
    config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream({"messages": [("user", user_input)]}, config=config):
        for value in event.values():
            response = value["messages"][-1].content
            print("\n[bold green]Assistant:[/bold green]", response)

# -----------------------------------------------------------------------------
# 10) Função para uso direto da ferramenta (opcional)
# -----------------------------------------------------------------------------
def use_tool(user_query: str):
    """
    Exemplo de função que chama explicitamente a ferramenta TavilySearchResults,
    sem passar pelo fluxo principal do grafo. Mantida como referência.
    """
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
    print("\n[bold cyan]Chatbot (com memória) iniciado![/bold cyan]")
    print("[yellow]Digite 'quit', 'exit' ou 'q' para sair.[/yellow]")
    print("[yellow]Conversas persistem usando o thread_id='1' por padrão.[/yellow]\n")

    while True:
        try:
            user_input = input("[bold blue]User:[/bold blue] ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("[bold magenta]Até mais![/bold magenta]")
                break

            # Se a entrada começar com "search:", chamamos a ferramenta direto (caso queira)
            if user_input.lower().startswith("search:"):
                query = user_input.split("search:", 1)[1].strip()
                use_tool(query)
            else:
                # Caso contrário, processamos no grafo com memória
                stream_graph_updates(user_input, thread_id="1")

        except KeyboardInterrupt:
            print("\n[bold red]Chat encerrado pelo usuário.[/bold red]")
            break
        except Exception as e:
            print(f"\n[bold red]Erro inesperado:[/bold red] {e}")
