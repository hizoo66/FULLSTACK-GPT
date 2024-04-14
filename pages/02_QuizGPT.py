import json
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

import streamlit as st


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
)

with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        print(st.session_state["api_key"])

    if st.button(":red[Reset API_KEY]"):
        reset_api_key()

    api_key = api_key_input.text_input(
        ":blue[OpenAI API_KEY]",
        value=st.session_state["api_key"],
        key="api_key_input",
    )

    if api_key != st.session_state["api_key"]:
        st.session_state["api_key"] = api_key
        st.rerun()

    github_url = st.text("https://github.com/hizoo66/FULLSTACK-GPT/blob/main/pages/01_DocumentGPT.py")
    app_url = st.text("https://fullstack-gpt-8ewig3twyrqpwf5kis6bet.streamlit.app/DocumentGPT")
    maker = st.text("made by Hizoo")

if not api_key:
    st.warning("APIKEY를 입력해주세요!")


if "quiz_subject" not in st.session_state:
    st.session_state["quiz_subject"] = ""

if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False

def set_quiz_submitted(value: bool):
    st.session_state.update({"quiz_submitted": value})

@st.cache_data(show_spinner="퀴즈를 맛있게 굽고 있어요...")
def run_quiz_chain(*, subject, count, difficulty):
    chain = prompt | llm

    return chain.invoke(
        {
            "subject": subject,
            "count": count,
            "difficulty": difficulty,
        }
    )

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown(
            """
        #### 자~ 이제 퀴즈를 만들어 볼까요?
        """
         )
with col2:

    def reset_quiz():
        st.session_state["quiz_subject"] = ""
        run_quiz_chain.clear()

with st.form("quiz_create_form"):

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        quiz_subject = st.text_input(
            ":blue[주제]",
            placeholder="무엇을 주제로 퀴즈를 만들까요?",
            value=st.session_state["quiz_subject"],
        )

    with col2:
        quiz_count = st.number_input(
            ":blue[개수]",
            placeholder="개수",
            value=10,
            min_value=2,
            )

    with col3:
        quiz_difficulty = st.selectbox(
            ":blue[레벨]",
            ["1", "2", "3", "4", "5"],
            )

    st.form_submit_button(
        "**:blue[퀴즈 만들기 시작]**",
        use_container_width=True,
        on_click=set_quiz_submitted,
        args=(False,),
    )

    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

prompt = PromptTemplate.from_template(
            """            
            Please create a quiz based on the following criteria:

            Topic: {subject}
            Number of Questions: {count}
            Difficulty Level: Level-{difficulty}/5
            Language: Korean

            The quiz should be well-structured with clear questions and correct answers.
            Ensure that the questions are relevant to the specified topic and adhere to the selected difficulty level.
            The quiz format should be multiple-choice,
            and each question should be accompanied by four possible answers, with only one correct option.
            """,
        )

if quiz_subject:
    response_box = st.empty()
    response = run_quiz_chain(
        subject=quiz_subject,
        count=quiz_count,
        difficulty=quiz_difficulty,
    )
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)

    generated_quiz_count = len(response["questions"])

    with st.form("quiz_questions_form"):
        solved_count = 0
        correct_count = 0
        answer_feedback_box = []
        answer_feedback_content = []

        for index, question in enumerate(response["questions"]):
            st.write(f"{index+1}. {question['question']}")
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                label_visibility="collapsed",
                key=f"[{quiz_subject}_{quiz_count}_{quiz_difficulty}]question_{index}",
            )

            answer_feedback = st.empty()
            answer_feedback_box.append(answer_feedback)

            if value:
                solved_count += 1

                if {"answer": value, "correct": True} in question["answers"]:
                    answer_feedback_content.append(
                        {
                            "index": index,
                            "correct": True,
                            "feedback": "정답! :100:",
                        }
                    )
                    correct_count += 1
                else:
                    answer_feedback_content.append(
                        {
                            "index": index,
                            "correct": False,
                            "feedback": "다시 도전해 보아요! :sparkles:",
                        }
                    )
                    
        is_quiz_all_submitted = solved_count == generated_quiz_count

        if is_quiz_all_submitted:
            for answer_feedback in answer_feedback_content:
                index = answer_feedback["index"]
                with answer_feedback_box[index]:
                            if answer_feedback["correct"]:
                                st.success(answer_feedback["feedback"])
                            else:
                                st.error(answer_feedback["feedback"])

                st.divider()

            if correct_count == generated_quiz_count:
                for _ in range(3):
                    st.balloons()

with st.sidebar:
    github_url = st.text("https://github.com/hizoo66/FULLSTACK-GPT/blob/main/pages/02_QuizGPT.py")
    app_url = st.text("https://fullstack-gpt-8ewig3twyrqpwf5kis6bet.streamlit.app/QuizGPT")
    maker = st.text("made by Hizoo")