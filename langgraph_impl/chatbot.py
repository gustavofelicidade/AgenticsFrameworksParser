from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from rich import print
from rich.pretty import Pretty
import time


# Define o estado do chatbot
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Inicializar o gráfico
graph_builder = StateGraph(State)

# Configuração do modelo LLM
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm = ChatOpenAI()

# Função do chatbot
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# Adicionar o nó "chatbot"
graph_builder.add_node("chatbot", chatbot)

# Adicionar os pontos de entrada e saída
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compilar o gráfico
graph = graph_builder.compile()


# Visualizar o gráfico com Matplotlib (ASCII)
def visualize_graph_ascii(graph):
    try:
        # Renderizar o gráfico em ASCII
        ascii_graph = graph.get_graph().draw_ascii()
        print("\n[bold blue]Gráfico do Chatbot (Formato ASCII):[/bold blue]")
        print(ascii_graph)  # Imprimir no console
    except Exception as e:
        print("Erro ao renderizar o gráfico em ASCII:", e)


# Fluxo de mensagens do chatbot
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            # Acessar diretamente o atributo `content` do objeto
            response = value["messages"][-1].content
            print("\n[bold green]Assistant:[/bold green]", response)


# Exemplo de execução
if __name__ == "__main__":
    # Visualizar o gráfico
    visualize_graph_ascii(graph)

    # Loop interativo do chatbot
    print("\n[bold cyan]Chatbot iniciado! Digite sua mensagem.[/bold cyan]")
    print("[yellow]Digite 'quit', 'exit' ou 'q' para sair.[/yellow]\n")

    while True:
        try:
            user_input = input("[bold blue]User:[/bold blue] ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("[bold magenta]Até mais![/bold magenta]")
                break

            # Simular barra de progresso durante o processamento
            print("\n[cyan]Processando sua mensagem...[/cyan]")
            for _ in tqdm(range(10), desc="Progresso", bar_format="{l_bar}{bar} [tempo restante: {remaining}]"):
                time.sleep(0.1)

            # Processar resposta do chatbot
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\n[bold red]Chat encerrado pelo usuário.[/bold red]")
            break
        except Exception as e:
            print(f"\n[bold red]Erro inesperado:[/bold red] {e}")
