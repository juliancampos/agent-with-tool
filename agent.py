from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

import os

# 1 - Carrega API KEY 
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 2 - Definição do modelo
llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=API_KEY)

# 3 - Define o prompt do sistema
system_message = SystemMessage(content="""
    Você é um assistente inteligente.
    Se o Usuário pedir contas, use a ferramenta 'sum'.
    Caso contrário, apenas responda normalmente.
""")

# 4 - Definindo ferramenta de soma
@tool("sum")
def sum(valores: str) -> str:
    """Soma números separados por vírgula."""
    try:
        a, b = map(float, valores.split(","))
        return str(a + b)
    except Exception as e:
        return f"Erro ao somar os valores: {e}"

# 5 - Criação do Agente com LangGraph
tools = [sum]

graph = create_react_agent(
    model=llm_model,
    tools=tools,
    prompt=system_message
)
export_graph = graph

# 6 -  Extrair resposta final
def final_response(result):
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        return ai_messages[-1].content
    else:
        return "Nenhuma resposta gerada pelo modelo."

# 7 - Executando o agente
if __name__ == "__main__":
    firstMessage = HumanMessage(content="Qual é a soma de 15 e 27?")
    firstResult = export_graph.invoke({"messages": [firstMessage]})
    finalFirstAnswer = final_response(firstResult)
    print("Resposta do modelo:", finalFirstAnswer)

    print("================================")

    secondMessage = HumanMessage(content="Vale a pena estudar Golang ?")
    secondResult = export_graph.invoke({"messages": [secondMessage]})
    finalSecondAnswer = final_response(secondResult)
    print("Resposta do modelo:", finalSecondAnswer)
