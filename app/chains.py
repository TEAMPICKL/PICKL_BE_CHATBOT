from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from base import BaseChain

class TopicChain(BaseChain):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Explain the given topic concisely. Answer in Korean."
        )

    def setup(self):
        llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", "Here is the topic: {topic}")]
        )
        return prompt | llm | StrOutputParser()


class ChatChain(BaseChain):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or (
            "You are a helpful AI Assistant. Your name is '테디'. You must answer in Korean."
        )

    def setup(self):
        llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), MessagesPlaceholder(variable_name="messages")]
        )
        return prompt | llm | StrOutputParser()


class LLM(BaseChain):
    def setup(self):
        return ChatOpenAI(model=self.model, temperature=self.temperature)


class Translator(BaseChain):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Translate the given sentences into Korean."
        )

    def setup(self):
        llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", "Here is the sentence: {input}")]
        )
        return prompt | llm | StrOutputParser()