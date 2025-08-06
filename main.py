from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="tinyllama")

template = """you are a helpful assistant. Answer the question based on the context provided.
question: {question}
review: {reviews}"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------------------------------")
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"question": question, "reviews": reviews})
    print("\n---------------------------------------------------")

    print(result)