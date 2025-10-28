# fixed_example.py
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Build the chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful customer support agent."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(
        "Customer issue: {issue_description}"
    ),
])

# Load chat history from file and convert to message objects.
# Here we treat each line as a human message for simplicity.
chat_history = []
with open("chat_history.txt", "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        if not text:
            continue
        # If your file contains role info (like "user: Hi"), parse and create AI/Human messages accordingly.
        # For now we wrap each line as a HumanMessage.
        chat_history.append(HumanMessage(content=text))

print("Chat history (as Message objects):", chat_history)

# --- Option A: get PromptValue and then messages (recommended when you need structured messages) ---
prompt_value = chat_prompt_template.format_prompt(
    chat_history=chat_history,
    issue_description="My refund status?"
)

# prompt_value is a ChatPromptValue / PromptValue; now convert to messages
messages = prompt_value.to_messages()
print("\nmessages from format_prompt().to_messages()")
for m in messages:
    print(type(m).__name__, ":", getattr(m, "content", str(m)))

# --- Option B: get a formatted string if you want plain text ---
formatted_string = chat_prompt_template.format(
    chat_history=chat_history,
    issue_description="My refund status?"
)
print("\nformatted string from format()")
print(formatted_string)
