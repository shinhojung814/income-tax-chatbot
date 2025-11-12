import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관한 질문을 답변해드립니다.")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def get_ai_message(user_message):
    # 1. 임베딩 설정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    # 2. 백터 데이터베이스 설정
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

    # 3. LLM 설정
    llm = ChatOpenAI(model='gpt-4o')

    # 4. 답변 생성용 프롬프트 설정
    answer_prompt = ChatPromptTemplate.from_template("""[Identity]
    - 당신은 한국 소득세 법의 전문가입니다.
    - [Context]를 참고해서 사용자의 질문에 답변해주세요.

    [Context]
    {context}

    질문: {question}

    Answer:""")

    # 5. Retriever 설정
    retriever = database.as_retriever(search_kwargs={"k": 4})

    # 6. QA 체인 설정
    qa_chain = (
        {
            "context": lambda x: "\n\n".join(doc.page_content for doc in retriever.invoke(x["query"])),
            "question": lambda x: x["query"]
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    # 7. 사전 정의
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # 8. 질문 변환용 프롬프트 (사전 적용)
    dictionary_prompt = ChatPromptTemplate.from_template("""당신은 한국 소득세 법의 전문가입니다.

    사용자의 질문을 읽고, 아래 사전을 참고해서 질문을 더 정확한 법률 용어로 변경해주세요.
    만약 변경할 필요가 없다고 판단되면 원래 질문을 그대로 반환해주세요.

    사전: {dictionary}

    기존 질문: {question}

    변경된 질문:""")

    # 9. 질문 변환 체인 (사전 적용)
    dictionary_chain = (
        {
            "dictionary": lambda _: dictionary,
            "question": lambda x: x["question"]
        }
        |dictionary_prompt
        | llm
        | StrOutputParser()
    )

    # 10. 소득세 챗봇 체인 설정
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message

if user_question := st.chat_input(placeholder="소득세에 관한 질문을 입력해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다."):
        ai_message = get_ai_message(user_question)

        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})