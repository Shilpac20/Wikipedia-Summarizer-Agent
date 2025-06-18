import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import WikipediaAPIWrapper

# ------------------ Model Options ------------------
MODEL_OPTIONS = {
    "Phi-2 (Microsoft)": "microsoft/phi-2",
    "TinyLLaMA 1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

# ------------------ Load LLM ------------------
@st.cache_resource
def load_llm(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        truncation=True,
        temperature=0.3,
        return_full_text=False,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# ------------------ Wikipedia Tool ------------------
def wikipedia_with_link(query: str) -> str:
    wiki = WikipediaAPIWrapper(lang="en", top_k_results=1, doc_content_chars_max=1000)
    summary = wiki.run(query)

    match = re.search(r"Page:\s*(.+?)\s*Summary:\s*(.+)", summary, re.DOTALL)
    if match:
        page = match.group(1).strip()
        text = match.group(2).strip()
        url = f"https://en.wikipedia.org/wiki/{page.replace(' ', '_')}"
        return f"{text}\n\n🔗 [Read more on Wikipedia]({url})"
    else:
        return summary

# ------------------ Rephraser Tool ------------------
def make_rephrase_tool(llm):
    def rephrase(text: str) -> str:
        prompt = f"""Rephrase the following passage in a simple, friendly, and clear way. 
Make it easy to understand for someone without technical background. Keep the important facts, and avoid adding extra information.

Text:
{text}

Rephrased version:"""
        return llm(prompt)

    return Tool(
        name="Rephraser",
        func=rephrase,
        description="Simplifies and rephrases complex text into friendly language."
    )

# ------------------ Tool List ------------------
def get_tools(llm):
    return [
        Tool(
            name="Wikipedia",
            func=wikipedia_with_link,
            description="Search Wikipedia and return a summary with a link to the article."
        ),
        make_rephrase_tool(llm)
    ]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🧠 Wikipedia Summarizer Agent", layout="centered")
st.title("🧠 Wikipedia Summarizer Agent (with LangChain)")

model_name = st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[model_name]

query = st.text_input("Ask a question:", placeholder="e.g., Who is C.V. Raman?")

if query:
    with st.spinner(f"Loading {model_name}..."):
        llm = load_llm(model_id)
        tools = get_tools(llm)

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True , # Set to False if you don't want to see internal reasoning
            handle_parsing_errors=True
        )

    with st.spinner("Agent is thinking..."):
        response = agent.run(query)

    st.success("Agent Response:")
    st.markdown(response)
