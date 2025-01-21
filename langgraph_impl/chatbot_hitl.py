"""
chatbot_hitl.py
================

Este módulo demonstra como introduzir um "humano no circuito" (Human-in-the-Loop, HITL)
em um chatbot construído com LangGraph, aproveitando o conceito de interrupção
(`interrupt`) para pausar a execução e aguardar input humano antes de continuar.

Esta versão inclui o `interrupt_before=["tools"]` na compilação do grafo,
permitindo que o fluxo seja interrompido antes de executar qualquer ferramenta.
Ao retomar, podemos verificar ou alterar a chamada de ferramenta, ou simplesmente
dar continuidade, sempre mantendo um loop de interação com o usuário (como
nas partes 1, 2 e 3 do tutorial).

Utilizamos a biblioteca `rich` para imprimir as mensagens de maneira mais
legível. Caso sua versão do LangChain/Graph não possua `role` nas mensagens,
o código verifica também a propriedade `type`.

Requisitos mínimos:
- langchain_anthropic
- langgraph (com o módulo `checkpoint.memory` e `prebuilt`)
- langchain_community (para TavilySearchResults)
- langchain_core (para o decorador `tool`)
- tqdm, matplotlib, rich, pprint (opcional, para logs e visualização)
- (Opcional) `grandalf` para desenhar o gráfico em ASCII se desejado

Autor: Seu Nome
"""
import os
import time
import json
import matplotlib.pyplot as plt
from typing import Annotated
from typing_extensions import TypedDict
from pprint import pprint
from tqdm import tqdm

# Pacotes para exibição colorida e legível de logs
from rich import print
from rich.console import Console

# --------------------------------------------------------------------------
# 1) Importações específicas do LangGraph e da ferramenta Tavily
# --------------------------------------------------------------------------
# from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Novo: Import do MemorySaver para manter histórico em memória
# e import do interrupt para permitir pausa e retomada.
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Classes e métodos do LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

console = Console()

# -----------------------------------------------------------------------------
# 2) Definição de State e criação do "graph_builder"
# -----------------------------------------------------------------------------
# Nosso estado (State) carrega a lista de mensagens trocadas.

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# -----------------------------------------------------------------------------
# 3) Ferramenta de assistência humana
# -----------------------------------------------------------------------------
# Decorador @tool que transforma a função em uma ferramenta invocável
# pelo LLM. Internamente, chamamos `interrupt()`, que sinaliza
# ao LangGraph para pausar a execução e aguardar intervenção humana.

@tool
def human_assistance(query: str) -> str:
    """
    Solicita assistência de um humano.
    A execução é interrompida até que o humano insira uma resposta.
    """
    human_response = interrupt({"query": query})
    return human_response["data"]

# -----------------------------------------------------------------------------
# 4) Configuração do LLM e demais ferramentas
# -----------------------------------------------------------------------------
tool_search = TavilySearchResults(max_results=2)
tools = [tool_search, human_assistance]

# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)


# -----------------------------------------------------------------------------
# 5) Nó chatbot
# -----------------------------------------------------------------------------
# Aqui adicionamos um 'assert' para garantir que o LLM não faça múltiplas
# chamadas de ferramenta em paralelo, o que poderia complicar a retomada.

def chatbot(state: State):
    """
    Nó principal do chatbot, que chama o llm_with_tools.
    Caso o LLM gere chamadas para ferramentas (tool_calls), elas serão
    tratadas posteriormente pelo ToolNode.
    """
    message = llm_with_tools.invoke(state["messages"])
    # Garante que não haja mais de uma chamada de ferramenta em paralelo
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

# -----------------------------------------------------------------------------
# 6) Nó de ferramentas (ToolNode)
# -----------------------------------------------------------------------------
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# -----------------------------------------------------------------------------
# 7) Roteamento condicional (tools_condition)
# -----------------------------------------------------------------------------
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Se a mensagem do LLM contiver uma tool_call, vamos ao nó "tools"
graph_builder.add_edge("tools", "chatbot")
# O fluxo começa no "chatbot"
graph_builder.add_edge(START, "chatbot")

# -----------------------------------------------------------------------------
# 8) Configuração do checkpointer e compilação (com interrupt_before=["tools"])
# -----------------------------------------------------------------------------
memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # Interrompe antes de rodar o nó "tools"
    # interrupt_after=["tools"] # Exemplo de interrupção depois, se desejado
)

# -----------------------------------------------------------------------------
# 9) Função de visualização do gráfico em ASCII (opcional)
# -----------------------------------------------------------------------------
def visualize_graph_ascii(graph):
    """
    Tenta renderizar o gráfico em formato ASCII e imprimir no console.
    Pode falhar caso não haja suporte local ou dependências de visualização.
    """
    try:
        ascii_graph = graph.get_graph().draw_ascii()
        print("\n[bold blue]Gráfico do Chatbot (Formato ASCII):[/bold blue]")
        print(ascii_graph)
    except Exception as e:
        print("Erro ao renderizar o gráfico em ASCII:", e)

# -----------------------------------------------------------------------------
# 10) Função de fluxo de mensagens, lidando com a interrupção
# -----------------------------------------------------------------------------
def stream_graph_updates(user_input: str, thread_id: str = "1"):
    """
    Executa o grafo usando thread_id para persistir a memória.
    Caso o fluxo seja interrompido antes de 'tools', perguntamos ao usuário
    se deseja prosseguir ou não. Se prosseguir, chamamos `graph.stream(None)`.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Primeira fase: enviamos a mensagem do usuário
    events = list(graph.stream({"messages": [("user", user_input)]}, config=config, stream_mode="values"))

    # Processamos as mensagens retornadas
    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
            # Verifica se a mensagem tem role ou type
            msg_role = getattr(last_msg, "role", None) or getattr(last_msg, "type", None)
            msg_content = last_msg.content

            if msg_role == "assistant":
                console.print(f"\n[green]Assistant:[/green] {msg_content}", style="bold")
            elif msg_role == "tool":
                console.print(f"[magenta]Tool Message:[/magenta] {msg_content}", style="bold")
            elif msg_role == "user":
                console.print(f"\n[yellow]User:[/yellow] {msg_content}", style="bold")
            else:
                console.print(f"\n[bold]{msg_role or 'unknown'}:[/bold] {msg_content}")

    # Agora verificamos se o fluxo parou antes de executar 'tools'.
    snapshot = graph.get_state(config)
    if snapshot.next and "tools" in snapshot.next:
        console.print("\n[bold red]Interrupção ativada antes de 'tools'![/bold red]")
        # Podemos inspecionar a última mensagem para ver as tool_calls
        existing_message = snapshot.values["messages"][-1]
        tool_calls = getattr(existing_message, "tool_calls", [])
        if tool_calls:
            console.print("\n[white bold]Chamadas de Ferramenta Detectadas:[/white bold]")
            pprint(tool_calls)
            console.print("[yellow]Deseja prosseguir com a ferramenta? (s/n)[/yellow]")
            choice = input("> ").strip().lower()
            if choice.startswith("s"):
                console.print("[green]Prosseguindo...[/green]")
                events2 = list(graph.stream(None, config=config, stream_mode="values"))
                # Exibimos as mensagens finais após retomar
                for ev in events2:
                    if "messages" in ev:
                        last_msg = ev["messages"][-1]
                        msg_role = getattr(last_msg, "role", None) or getattr(last_msg, "type", None)
                        msg_content = last_msg.content

                        if msg_role == "assistant":
                            console.print(f"\n[green]Assistant:[/green] {msg_content}", style="bold")
                        elif msg_role == "tool":
                            console.print(f"[magenta]Tool Message:[/magenta] {msg_content}", style="bold")
                        else:
                            console.print(f"\n[bold]{msg_role or 'unknown'}:[/bold] {msg_content}")
            else:
                console.print("[red]Ferramenta abortada. Não iremos retomar.[/red]")
        else:
            console.print("[dim]Nenhum tool_call encontrado. Nada para aprovar.[/dim]")

# -----------------------------------------------------------------------------
# 11) Execução principal (main)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Visualizar o grafo no console (em formato ASCII)
    visualize_graph_ascii(graph)

    # Loop de interação com o usuário via terminal
    console.print("\n[bold cyan]Chatbot com HITL e interrupção antes de 'tools' iniciado![/bold cyan]")
    console.print("[yellow]Digite 'quit', 'exit' ou 'q' para sair.[/yellow]")
    console.print("[yellow]Cada vez que o LLM quiser chamar uma ferramenta, perguntaremos se deseja prosseguir.\n[/yellow]")

    while True:
        try:
            user_input = input("[bold blue]User:[/bold blue] ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[bold magenta]Até mais![/bold magenta]")
                break
            stream_graph_updates(user_input, thread_id="1")

        except KeyboardInterrupt:
            console.print("\n[bold red]Chat encerrado pelo usuário.[/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Erro inesperado:[/bold red] {e}")
