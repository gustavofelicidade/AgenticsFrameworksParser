"""
06_chatbot_time_travel.py
=========================

Este módulo demonstra como realizar "viagem no tempo" em um fluxo de trabalho
de chatbot utilizando LangGraph. A viagem no tempo permite que voltemos a um
estado anterior do grafo e retomemos a execução a partir daquele ponto,
possibilitando experimentar ramificações diferentes ou corrigir erros
sem perder o histórico de interações.

Baseado na Parte 6 do tutorial, este script mostra:
- Como configurar um chatbot simples com ferramentas (usando TavilySearchResults).
- Como salvar o estado das interações com MemorySaver.
- Como listar o histórico de estados salvos (checkpoints) via get_state_history.
- Como retomar a execução a partir de um checkpoint específico.

Requisitos mínimos:
- langchain_openai (para ChatOpenAI)
- langgraph (com o módulo checkpoint.memory e prebuilt)
- langchain_community (para TavilySearchResults)
- langchain_core (para representação de mensagens)
- tqdm, matplotlib, rich, pprint (opcionais, para logs e visualização)
- (Opcional) grandalf para desenhar o gráfico em ASCII se desejado

Autor: Seu Nome
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

# --------------------------------------------------------------------------
# Importações específicas do LangGraph e da ferramenta Tavily
# --------------------------------------------------------------------------
from langchain_openai import ChatOpenAI  # LLM substituindo ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage  # Opcional, caso precise manipular mensagens

# Novo: Import do MemorySaver para manter o histórico (checkpointing)
from langgraph.checkpoint.memory import MemorySaver

# Classes e métodos do LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --------------------------------------------------------------------------
# 1) Definição do estado (State) com a lista de mensagens
# --------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Criamos um construtor de grafo com esse estado
graph_builder = StateGraph(State)

# --------------------------------------------------------------------------
# 2) Configuração do LLM e ferramenta
# --------------------------------------------------------------------------
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Substituindo ChatAnthropic por ChatOpenAI
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

# --------------------------------------------------------------------------
# 3) Nó principal do chatbot
# --------------------------------------------------------------------------
def chatbot(state: State):
    """
    Nó de processamento que aciona o LLM (ChatOpenAI) com ferramentas vinculadas.
    Retorna a nova mensagem em um dicionário com a chave 'messages'.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# --------------------------------------------------------------------------
# 4) Nó de ferramentas (ToolNode)
# --------------------------------------------------------------------------
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# --------------------------------------------------------------------------
# 5) Roteamento condicional
# --------------------------------------------------------------------------
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# --------------------------------------------------------------------------
# 6) Configuração do checkpointer (MemorySaver) e compilação do grafo
# --------------------------------------------------------------------------
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# --------------------------------------------------------------------------
# 7) Função auxiliar para exibir o grafo em ASCII (opcional)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 8) Função para executar o grafo e imprimir mensagens na tela
# --------------------------------------------------------------------------
def run_chatbot(messages_input, config, description=""):
    """
    Executa o grafo com uma determinada entrada de mensagens.
    As saídas (mensagens do assistente e tool_messages) são impressas na tela.
    """
    print(f"\n[bold cyan]--- {description} ---[/bold cyan]") if description else None

    events = graph.stream(messages_input, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            # Pegamos a última mensagem retornada
            last_msg = event["messages"][-1]
            role = getattr(last_msg, "role", None) or getattr(last_msg, "type", None)
            content = last_msg.content
            if role == "user":
                print("\n[bold yellow]User:[/bold yellow]", content)
            elif role == "assistant":
                print("\n[bold green]Assistant:[/bold green]", content)
            elif role == "tool":
                print("\n[bold magenta]Tool Message:[/bold magenta]", content)
            else:
                print(f"\n[bold]{role or 'unknown'}:[/bold]", content)

# --------------------------------------------------------------------------
# 9) Demonstração principal de "viagem no tempo"
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Visualização opcional do grafo
    visualize_graph_ascii(graph)

    # Config para usar thread_id='1'
    config = {"configurable": {"thread_id": "1"}}

    # Primeira interação
    user_prompt_1 = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Estou assistindo BBB25. Pode fazer uma pesquisa pra mim?"
                ),
            }
        ]
    }
    run_chatbot(user_prompt_1, config, description="Passo 1")

    # Segunda interação
    user_prompt_2 = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll build an autonomous agent with it!"
                ),
            }
        ]
    }
    run_chatbot(user_prompt_2, config, description="Passo 2")

    # 10) Listar o histórico de estados e escolher um para "rebobinar"
    print("\n[bold white]Exibindo histórico de checkpoints...[/bold white]")
    all_states = graph.get_state_history(config)
    to_replay = None

    for state in all_states:
        msg_count = len(state.values["messages"])
        nxt = state.next
        print(f"Num Messages: {msg_count} | Next: {nxt}")
        print("-" * 80)

        # Exemplo: vamos escolher algum estado cuja contagem de mensagens == 6
        # (valor arbitrário conforme tutorial).
        if msg_count == 6:
            to_replay = state

    # 11) Se encontrarmos um estado para retomar, usamos to_replay.config
    if to_replay is not None:
        print(
            f"\n[bold cyan]--- Retomando a partir do estado com checkpoint_id={to_replay.config.get('checkpoint_id', '')} ---[/bold cyan]"
        )
        # Chamamos o grafo novamente com stream(None, config_do_checkpoint)
        # Observação: no tutorial, .config está aninhado:
        # 'checkpoint_ns': '', 'checkpoint_id': '...'
        # É esse checkpoint_id que diz ao checkpointer qual estado restaurar.
        run_chatbot(None, to_replay.config, description="Retomando do checkpoint")

    else:
        print("\n[bold red]Nenhum estado para replay encontrado.[/bold red]")

    print("\n[bold magenta]Demonstração de viagem no tempo finalizada![/bold magenta]")
