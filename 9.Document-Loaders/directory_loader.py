from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs=loader.load() 
# #loads all the pages which will very costly and time consuming
docs = loader.lazy_load() # its solves it 

for document in docs:
    print(document.metadata)

# print(docs[0].page_content)

# print(docs[23].metadata)