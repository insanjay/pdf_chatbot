from rag_core import ask_question

print("PDF Chatbot Ready")

while True:

    query = input("You: ")

    if query == "exit":
        break

    answer = ask_question(query)

    print("Bot:", answer)