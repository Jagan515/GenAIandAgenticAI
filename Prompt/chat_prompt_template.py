from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from proto import Message




#Wrong way to create chat prompt template
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a helpful {domain} expert that provides concise and accurate information."),
#     HumanMessage(content="Explain in simple English what {topic} is.")
# ])

chat_template = ChatPromptTemplate.from_messages((
    SystemMessagePromptTemplate.from_template(
        "You are a helpful {domain} expert that provides concise and accurate information."
    ),
    HumanMessagePromptTemplate.from_template(
        "Explain in simple English what {topic} is."
    )
))

# produce the prompt with variables
chat_value = chat_template.format_prompt(domain="science", topic="quantum computing")

# get the sequence of messages ready to send to the model
messages = chat_value.to_messages()
print("Messages:", messages)