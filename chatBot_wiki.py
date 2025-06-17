import streamlit as st
import re
import torch
import wikipedia
from langchain.utilities import WikipediaAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ Model Selector ------------------
MODEL_OPTIONS = {
    "Phi-2 (Microsoft)": "microsoft/phi-2",
    "TinyLLaMA 1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

# ------------------ Load LLM ------------------
@st.cache_resource(show_spinner=True)
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
        do_sample=False,  # Deterministic output to avoid hallucinations
        temperature=0.3,
        top_p=0.8
    )
    return pipe

# ------------------ Get Wikipedia Summary ------------------
class WikipediaToolWithURL:
    def __init__(self):
        self.api = WikipediaAPIWrapper(lang="en", top_k_results=1, doc_content_chars_max=1000)
        wikipedia.set_lang("en")

    def run(self, query: str) -> dict:
        summary = self.api.run(query)
        try:
            if summary:
                match = re.search(r"Page:\s*(.+?)\s*Summary:\s*(.+)", summary, re.DOTALL)
                if match:
                    page = match.group(1).strip()
                    summary_gist = match.group(2).strip()
                    page_url = f"https://en.wikipedia.org/wiki/{page.replace(' ', '_')}"
                    return {
                        "summary": summary_gist,
                        "link": f"🔗 Read more on Wikipedia Page - [{page}]({page_url})"
                    }
        except Exception as e:
            print(f"[Wikipedia URL Error] {e}")
        return {"summary": summary or "No summary found.", "link": ""}

# ------------------ Rephrase Summary ------------------
def rephrase_summary(pipe, summary: str):
    prompt = (
        "You are a helpful assistant. Rephrase the following text in a simpler, friendly tone, "
        "without changing any facts or adding new information:\n\n"
        f"\"{summary}\"\n\nRephrased:"
    )
    result = pipe(prompt)[0]["generated_text"]
    rephrased = result.split("Rephrased:")[-1].strip()
    return rephrased, result

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="📚 Ask Wikipedia", layout="centered")
st.title("📚 Ask Wikipedia Agent")
st.markdown("Choose a model, enter a topic, and get a rephrased Wikipedia summary with the original link.")

model_label = st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[model_label]

query = st.text_input("Enter your topic:", placeholder="e.g., Quantum Mechanics")

if query:
    pipe = load_llm(model_id)
    wiki = WikipediaToolWithURL()
    with st.spinner("Getting Wikipedia summary..."):
        answer = wiki.run(query)
        summary = answer['summary']
        url = answer['link']

    if url:
        with st.spinner("Rephrasing summary with LLM..."):
            new_summary, raw_output = rephrase_summary(pipe, summary)

        #st.markdown(f"### 📄 Original Summary\n{summary}")
        st.success("### ✨ Summary")
        st.markdown(new_summary)
        st.markdown(url)

    else:
        st.error(summary)
