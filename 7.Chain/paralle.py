from json import load
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1=HuggingFaceEndpoint(
     repo_id="HuggingFaceTB/SmolLM3-3B",
     task="text-generation"
)
llm2=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model1=ChatHuggingFace(llm=llm1)
model2=ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

merge_chain= prompt3 | model2 | parser

chain= parallel_chain | merge_chain

text=""" 
    Natural Language Understanding (NLU) is a subfield of Natural Language Processing that provides machines with the ability to interpret and extract meaning from human language. It enables systems to understand the intent behind linguistic inputs. NLU serves as the foundation for a wide range of language-driven applications including chatbots, virtual assistants and content moderation systems.

nlg
NLU in the NLP landscape
As machines operate in binary code, the gap between language and machine-readable information requires language understanding. NLU helps this cause by applying statistical methods to analyze syntax, semantics and dependencies in text.

Key Models and Techniques in NLU
1. Transformers: Modern NLU is powered by transformer architectures that capture contextual relationships -

BERT: Uses bidirectional attention to understand sentence meaning.
T5: Treats every task as text-to-text, which simplifies fine-tuning.
GPT: Focuses on generating and understanding text in a conversational setting.
2. Recurrent Neural Networks (RNNs): RNNs analyze text sequentially and maintain context. Variants like LSTM and GRU handle long-term dependencies and improve stability.

3. Word Embeddings: Word2Vec and GloVe map words into dense vector spaces where similar meanings lie closer, helping machines reason about semantic similarity.

4. Rule-Based Systems: Useful for domain-specific systems, especially in the cases when predictable structure exists.

5. Conditional Random Fields (CRFs): CRFs are used in sequence labeling tasks such as POS tagging and NER, capturing dependencies between predicted labels.    
"""
response = chain.invoke({'text':text})

print(response)

chain.get_graph().print_ascii()


