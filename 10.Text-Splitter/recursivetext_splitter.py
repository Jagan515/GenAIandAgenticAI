from langchain_text_splitters import RecursiveCharacterTextSplitter # mostly uses which conserves the meaning
from langchain_community.document_loaders import PyPDFLoader
#it uses splitter optimstic approach
loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0, # overlapping of the text in two chunks ,to conserve the text meaning (use 10-20%)
)

result = splitter.split_documents(docs)

print(result[1].page_content)
