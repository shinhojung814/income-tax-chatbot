from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM 설정
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)

    return llm

# 임베딩, 백터 데이터베이스 설정 및 Retriever 반환
def get_retriever(index_name='tax-markdown-index'):
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={"k": 4})

    return retriever

# 7. 사전 정의 및 질문 변환용 프롬프트 반환
def get_dictionary_chain():
    llm = get_llm()

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    dictionary_prompt = ChatPromptTemplate.from_template("""당신은 한국 소득세 법의 전문가입니다.

    사용자의 질문을 읽고, 아래 사전을 참고해서 질문을 더 정확한 법률 용어로 변경해주세요.
    만약 변경할 필요가 없다고 판단되면 원래 질문을 그대로 반환해주세요.

    사전: {dictionary}

    기존 질문: {question}

    변경된 질문:""")

    dictionary_chain = (
        {
            "dictionary": lambda _: dictionary,
            "question": lambda x: x["question"]
        }
        |dictionary_prompt
        | llm
        | StrOutputParser()
    )
    return dictionary_chain

# 4. QA 체인 설정 및 답변 생성용 프롬프트 반환
def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()

    answer_prompt = ChatPromptTemplate.from_template("""[Identity]
    - 당신은 한국 소득세 법의 전문가입니다.
    - [Context]를 참고해서 사용자의 질문에 답변해주세요.

    [Context]
    {context}

    질문: {question}

    Answer:""")

    qa_chain = (
        {
            "context": lambda x: "\n\n".join(doc.page_content for doc in retriever.invoke(x["query"])),
            "question": lambda x: x["query"]
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

# 소득세 챗봇 체인 설정 및 답변 반환
def get_ai_message(user_message):
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()

    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message