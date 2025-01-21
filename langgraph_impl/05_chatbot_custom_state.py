"""
05_chatbot_custom_state.py
==========================

Este módulo demonstra como criar um estado personalizado em nosso chatbot,
adicionando chaves adicionais (por exemplo, 'name' e 'birthday') ao estado
para lidar com fluxos de trabalho mais complexos. Assim, informações específicas
podem ser armazenadas e recuperadas de forma simples em todo o grafo de estados.

Nesta versão, introduzimos a ferramenta 'human_assistance' que faz uso de
``interrupt`` para solicitar revisão humana e, ao aprovar ou corrigir
as informações, atualiza o estado com um objeto Command.

Também demonstramos como podemos, manualmente, atualizar o estado de qualquer
chave usando o método ``graph.update_state()``.

Requisitos mínimos:
- langchain_openai (substituindo langchain_anthropic)
- langgraph (com o módulo checkpoint.memory e prebuilt)
- langchain_community (para TavilySearchResults)
- langchain_core (para o decorador @tool, Command, etc)
- tqdm, matplotlib, rich, pprint (opcionais, para logs e visualização)
- (Opcional) grandalf para desenhar o gráfico em ASCII se desejado


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
from langchain_openai import ChatOpenAI  # Usando ChatOpenAI (não Anthropic)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

# Novo: Import do MemorySaver para manter histórico em memória
# e import do interrupt para permitir pausa e retomada,
# além de Command para atualizar o estado.
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Classes e métodos do LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

console = Console()

# -----------------------------------------------------------------------------
# 2) Definição de State: agora com 'messages', 'name' e 'birthday'
# -----------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

# Criamos o builder do grafo com nossa nova definição de estado
graph_builder = StateGraph(State)

# -----------------------------------------------------------------------------
# 3) Ferramenta de assistência humana com atualização de estado
# -----------------------------------------------------------------------------
@tool
def human_assistance(
    name: str,
    birthday: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """
    Solicita assistência de um humano para revisar as informações fornecidas.
    Faz uso de interrupt() para pausar a execução até que o humano responda.
    Em seguida, retorna um Command que atualiza o estado com os dados
    revisados ou confirmados.
    """
    human_response = interrupt(
        {
            "question": "Esta informação está correta?",
            "name": name,
            "birthday": birthday,
        }
    )
    # Se o humano disser que está correto, mantemos as informações.
    if human_response.get("correct", "").lower().startswith("s"):
        verified_name = name
        verified_birthday = birthday
        response = "Informações confirmadas pelo revisor humano."
    else:
        # Caso contrário, aceitamos as correções feitas pelo humano
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Revisor humano fez uma correção: {human_response}"

    # Construímos o objeto de atualização de estado
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }

    # Retornamos um Command para atualizar o estado dentro do próprio tool
    return Command(update=state_update)

# -----------------------------------------------------------------------------
# 4) Configuração do LLM e das demais ferramentas
# -----------------------------------------------------------------------------
tool_search = TavilySearchResults(max_results=2)
tools = [tool_search, human_assistance]

llm = ChatOpenAI()  # Substituindo ChatAnthropic por ChatOpenAI
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------------------------
# 5) Nó chatbot
# -----------------------------------------------------------------------------
# Aqui adicionamos um 'assert' para garantir que o LLM não faça múltiplas
# chamadas de ferramenta em paralelo, o que complicaria a retomada.
def chatbot(state: State):
    """
    Nó principal do chatbot, que chama o llm_with_tools.
    Se houver chamadas de ferramenta, o fluxo seguirá para o ToolNode.
    """
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1, "Múltiplas tool_calls detectadas!"
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
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
# O fluxo começa no "chatbot"
graph_builder.add_edge(START, "chatbot")

# -----------------------------------------------------------------------------
# 8) Configuração do checkpointer e compilação (sem interrupção específica aqui)
# -----------------------------------------------------------------------------
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory
)

# -----------------------------------------------------------------------------
# 9) Função opcional de visualização do gráfico em ASCII
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
# 10) Função de fluxo de mensagens, lidando com tool_calls
# -----------------------------------------------------------------------------
def stream_graph_updates(user_input: str, thread_id: str = "1"):
    """
    Executa o grafo usando thread_id para persistir a memória.
    Se houver tool_calls, o fluxo seguirá automaticamente para o nó de ferramentas.
    """
    config = {"configurable": {"thread_id": thread_id}}

    events = graph.stream({"messages": [{"role": "user", "content": user_input}]},
                          config=config,
                          stream_mode="values")

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

# -----------------------------------------------------------------------------
# 11) Demonstração de uso: manual update e fluxo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # (Opcional) Visualizar o grafo em ASCII
    visualize_graph_ascii(graph)

    console.print("\n[bold cyan]Chatbot com Estado Personalizado![/bold cyan]")
    console.print("[yellow]Digite 'quit', 'exit' ou 'q' para sair.\n[/yellow]")

    # Exemplo rápido de interação
    while True:
        try:
            user_input = input("[bold blue]User:[/bold blue] ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[bold magenta]Até mais![/bold magenta]")
                break

            # Enviamos a pergunta do usuário pelo grafo
            stream_graph_updates(user_input, thread_id="1")

            # Exemplo de como obter o snapshot do estado
            config = {"configurable": {"thread_id": "1"}}
            snapshot = graph.get_state(config)
            console.print("\n[dim]Estado atual (parcial):[/dim]")
            partial_state = {k: v for k, v in snapshot.values.items()
                             if k in ("name", "birthday")}
            pprint(partial_state)

            # Exemplo de como atualizar manualmente o estado
            console.print("\n[white]Deseja alterar manualmente o valor 'name'? (s/n)[/white]")
            choice = input("> ").strip().lower()
            if choice.startswith("s"):
                new_name = input("[white]Insira o novo valor para 'name':[/white] ")
                graph.update_state(config, {"name": new_name})
                console.print(f"[green]Estado atualizado com sucesso![/green]")
                # Verificando
                snapshot = graph.get_state(config)
                partial_state = {k: v for k, v in snapshot.values.items()
                                 if k in ("name", "birthday")}
                pprint(partial_state)

        except KeyboardInterrupt:
            console.print("\n[bold red]Chat encerrado pelo usuário.[/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Erro inesperado:[/bold red] {e}")
