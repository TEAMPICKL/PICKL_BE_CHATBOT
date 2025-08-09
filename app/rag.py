from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from base import BaseChain

def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata.get('page')}</page>"
        f"<source>{doc.metadata.get('source')}</source></document>"
        for doc in docs
    )

class RagChain(BaseChain):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or (
            "You are a helpful AI Assistant. Your name is '테디'. You must answer in Korean."
        )
        self.file_path = kwargs.get("file_path")

    def setup(self):
        if not self.file_path:
            raise ValueError("file_path is required")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = PDFPlumberLoader(self.file_path).load_and_split(text_splitter=splitter)

        # OpenAI 임베딩 권장: text-embedding-3-small (저렴/성능균형)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")
        llm = ChatOpenAI(model=self.model, temperature=self.temperature)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain