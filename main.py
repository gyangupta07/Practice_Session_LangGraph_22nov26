"""Streamlit UI for the notebook's AI candidate-screening workflow."""

from __future__ import annotations

import os

import streamlit as st
from langchain_openai import ChatOpenAI


st.set_page_config(
    page_title="Candidate Screening Assistant",
    page_icon="🧭",
    layout="centered",
)

st.title("Candidate Screening Assistant")
st.caption("Review an application against a Python Developer role.")

with st.sidebar:
    st.header("Configuration")
    st.caption("Provide an OpenAI API key to run the screening workflow.")
    try:
        saved_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        saved_api_key = ""
    api_key = st.text_input(
        "OpenAI API key",
        value=saved_api_key,
        type="password",
        help="You can also set OPENAI_API_KEY in your Streamlit secrets.",
    )
    st.divider()
    st.caption("The notebook uses gpt-4o-mini and screens for Python Developer skills.")

st.markdown("### Candidate application")
application = st.text_area(
    "Paste the application text",
    height=220,
    placeholder="For example: I have 5 years of software engineering experience, including Python...",
)


def screen_candidate(text: str, key: str) -> dict[str, str]:
    """Run the notebook's experience, skill-match, and routing steps."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)

    experience_prompt = (
        "Classify the candidate's experience from the application below. "
        "Return exactly one label: Entry-level, Mid-level, or Senior-level. "
        "Use Entry-level for 0-2 years, Mid-level for 3-5 years, and "
        "Senior-level for more than 5 years. If years are unclear, make the "
        "best supported choice. Do not add an explanation.\n\n"
        f"Application:\n{text}"
    )
    skill_prompt = (
        "Assess whether the application demonstrates skills relevant to a "
        "Python Developer role. Return exactly one label: Match or No Match. "
        "Do not add an explanation.\n\n"
        f"Application:\n{text}"
    )

    experience_raw = llm.invoke(experience_prompt).content
    skill_raw = llm.invoke(skill_prompt).content
    experience_text = str(experience_raw).strip()
    skill_text = str(skill_raw).strip()

    # Normalize model output so the workflow routes consistently.
    experience_lower = experience_text.lower()
    experience = next(
        (label for label in ("Senior-level", "Mid-level", "Entry-level")
         if label.lower() in experience_lower),
        "Unable to determine",
    )
    skill = "Match" if skill_text.lower().startswith("match") and not skill_text.lower().startswith("no match") else "No Match"

    if skill == "Match":
        response = "Candidate has been shortlisted for an HR interview."
        status = "Shortlisted"
    elif experience == "Senior-level":
        response = "Candidate has senior-level experience but doesn't match job skills."
        status = "Recruiter review"
    else:
        response = "Candidate doesn't meet JD and has been rejected."
        status = "Not shortlisted"

    return {
        "experience_level": experience,
        "skill_match": skill,
        "response": response,
        "status": status,
    }


if st.button("Screen application", type="primary", use_container_width=True):
    if not application.strip():
        st.warning("Paste an application before starting the screening.")
    elif not api_key.strip():
        st.warning("Add your OpenAI API key in the sidebar to run the screening.")
    else:
        try:
            with st.spinner("Reviewing the application…"):
                result = screen_candidate(application.strip(), api_key.strip())
            st.session_state["screening_result"] = result
            st.session_state["screening_application"] = application.strip()
        except Exception as exc:
            st.error(f"Screening could not be completed: {exc}")

if "screening_result" in st.session_state:
    result = st.session_state["screening_result"]
    st.markdown("### Screening result")
    st.success(result["status"])
    first, second = st.columns(2)
    first.metric("Experience level", result["experience_level"])
    second.metric("Python skill match", result["skill_match"])
    st.info(result["response"])
    with st.expander("Application reviewed"):
        st.write(st.session_state["screening_application"])

st.caption("AI output is a screening aid. Review applications and decisions before taking action.")
