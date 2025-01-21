"""
05_chatbot_custom_state.py
==========================

Este módulo demonstra como personalizar o estado (State) de um chatbot
construído com LangGraph, adicionando campos adicionais (por exemplo,
"name" e "birthday") além da lista de mensagens. Isso permite fluxos mais
avançados, onde podemos, por exemplo, pedir a validação humana antes de
armazenar (ou confirmar) tais informações.

Nesta parte 5 do tutorial, aproveitamos o conceito de "human in the loop"
de maneira que o próprio chatbot, ao encontrar uma possível resposta para
uma query (por exemplo, uma data de lançamento de um produto), solicite
confirmação humana antes de atualizar o estado global.

Utilizamos as seguintes classes e funções centrais:
- `StateGraph` para a construção do fluxo de nós (chatbot e ferramentas).
- `MemorySaver` para armazenar o histórico em memória.
- `ToolNode`, `tool` e `tools_condition` para lidar com as chamadas de ferramenta.
- `interrupt` para pausar a execução e pedir intervenção humana.
- `Command` para realizar ações de atualização de estado.

Requisitos mínimos:
- langchain_openai (no lugar de langchain_anthropic, conforme solicitação)
- langchain_community (para TavilySearchResults)
- langchain_core (para o decorador tool e tipos auxiliares)
- langgraph (com o módulo checkpoint.memory, prebuilt, etc.)
- tqdm, matplotlib, rich, pprint (opcional, para logs e visualização)
- (Opcional) grandalf para desenhar o gráfico em ASCII se desejado

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
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.tools import (
    tool,
    InjectedToolCallId,
)
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

console = Console()

# -----------------------------------------------------------------------------
# 2) Definição de State, incluindo agora name e birthday
# -----------------------------------------------------------------------------
# Nosso estado carrega a lista de mensagens E campos extras (name, birthday).

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

# -----------------------------------------------------------------------------
# 3) Ferramenta de assistência humana com atualização do estado
# -----------------------------------------------------------------------------
# Neste caso, solicitamos ao humano que confirme ou corrija "name" e "birthday".
# Se confirmado, mantemos os valores. Se corrigido, atualizamos a partir do input humano.
# Retornamos um objeto Command, que instruirá o LangGraph a atualizar o estado.

@tool
def human_assistance(
    name: str,
    birthday: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """
    Solicita assistência de um humano para verificar/corrigir informações.
    A execução é interrompida até que o humano insira se está correto (sim/não)
    e, se necessário, forneça correções.
    """
    human_response = interrupt(
        {
            "question": "Está correto?",
            "name": name,
            "birthday": birthday,
        }
    )
    # Verifica a resposta do humano
    if human_response.get("correct", "").lower().startswith("s"):  # s ou sim
        verified_name = name
        verified_birthday = birthday
        response = "Informação confirmada pelo humano."
    else:
        # Se o humano corrigir, usamos as informações fornecidas
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Informação corrigida: {human_response}"

    # Construímos o dicionário que será usado para atualizar o estado
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        # Registramos a mensagem da ferramenta (ToolMessage) no histórico
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }

    # Retornamos um Command com update, para LangGraph atualizar o estado
    return Command(update=state_update)

# -----------------------------------------------------------------------------
# 4) Configuração do LLM e demais ferramentas
# -----------------------------------------------------------------------------
tool_search = TavilySearchResults(max_results=2)
tools = [tool_search, human_assistance]

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------------------------
# 5) Nó chatbot (função principal de inferência)
# -----------------------------------------------------------------------------
def chatbot(state: State):
    """
    Nó principal do chatbot, que chama o llm_with_tools.
    Verifica se há apenas 0 ou 1 chamadas de ferramenta (para simplificar).
    """
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1, "Mais de uma chamada de ferramenta detectada!"
    return {"messages": [message]}

# -----------------------------------------------------------------------------
# 6) Construção do Grafo (StateGraph)
# -----------------------------------------------------------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# -----------------------------------------------------------------------------
# 7) Roteamento condicional (tools_condition)
# -----------------------------------------------------------------------------
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# -----------------------------------------------------------------------------
# 8) Checkpointer e compilação do Grafo
# -----------------------------------------------------------------------------
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# -----------------------------------------------------------------------------
# 9) Exemplo de uso e fluxo de execução
# -----------------------------------------------------------------------------
def run_example():
    """
    Função de demonstração:
    1) Usuário pergunta sobre a data de lançamento do LangGraph.
    2) O chatbot fará busca via TavilySearchResults.
    3) Em seguida, tenta preencher 'name' e 'birthday' e chama human_assistance.
    4) Interrompe para revisão humana (interrupt).
    5) Humano confirma ou corrige (resumimos com um Command).
    6) Estado final reflete os campos name e birthday confirmados/corrigidos.
    """
    user_input = (
        "Você pode descobrir quando o LangGraph foi lançado? "
        "Ao obter a resposta, use a ferramenta 'human_assistance' para revisão."
    )

    config = {"configurable": {"thread_id": "demo_custom_state"}}

    # Disparamos o fluxo inicial
    console.print("\n[bold cyan]=== Iniciando fluxo de chatbot ===[/bold cyan]")
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values"
    )

    # Exibimos as mensagens geradas até o ponto de interrupção (ou final)
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    console.print("\n[bold yellow]=== Interrupção para revisão humana! ===[/bold yellow]")
    console.print("[dim]Simulando a correção...[/dim]")

    # Simulando uma correção manual do humano
    # Exemplo: o humano diz "não está correto" e corrige para
    # name=LangGraph, birthday=17 de Janeiro de 2024
    human_command = Command(
        resume={
            "correct": "nao",
            "name": "LangGraph",
            "birthday": "17 de Janeiro de 2024"
        }
    )

    # Retomamos o fluxo após a interrupção com o 'human_command'
    events2 = graph.stream(human_command, config=config, stream_mode="values")
    for event in events2:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # Ao final, verificamos o estado persistido
    snapshot = graph.get_state(config)
    final_name = snapshot.values.get("name", "")
    final_birthday = snapshot.values.get("birthday", "")
    console.print(f"\n[bold green]Estado Final:[/bold green] name={final_name}, birthday={final_birthday}\n")

# -----------------------------------------------------------------------------
# 10) Execução principal (main)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Executa a demonstração de personalização do estado, criando um fluxo que
    pesquisa a data de lançamento de LangGraph e pede revisão humana.
    """
    run_example()
