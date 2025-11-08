# phishing_detector_final.py
"""
🕵️‍♂️ PHISHING EMAIL DETECTOR
CPU-friendly semantic phishing detector using SentenceTransformer (MiniLM),
with domain whitelist, keyword hints, and Gradio UI.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# ---------- 1. Config ----------
MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # lightweight model
device = "cpu"

# CSV path for whitelist
COMBINED_CSV = r"C:\Users\windows\Desktop\combined_110000 (1).csv"

# Load whitelist
if os.path.exists(COMBINED_CSV):
    df = pd.read_csv(COMBINED_CSV)
    SAFE_DOMAINS = set(df['text'].astype(str).str.strip().str.lower())
    print(f"🔐 Whitelisted domains loaded: {len(SAFE_DOMAINS)} entries")
else:
    SAFE_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com"}
    print(f"❌ CSV not found at {COMBINED_CSV}, using default whitelist")

# Suspicious keywords
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "password", "click", "confirm", "account",
    "suspended", "locked", "prize", "won", "payment", "refund"
]

# ---------- 2. URL / domain helpers ----------
URL_REGEX = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", flags=re.IGNORECASE)

def extract_urls(text: str):
    if not isinstance(text, str):
        return []
    return [u.rstrip('.,;!?)"\'') for u in URL_REGEX.findall(text)]

def domain_from_url(url: str):
    if not isinstance(url, str) or not url:
        return ""
    if not url.startswith(("http://", "https://")) and url.startswith("www."):
        url = "http://" + url
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def get_registered_domain(netloc: str):
    parts = [p for p in netloc.split('.') if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return netloc

# ---------- 3. Load SentenceTransformer ----------
print("⚡ Loading semantic model, this may take a few seconds...")
semantic_model = SentenceTransformer(MODEL_NAME, device=device)
print("✅ Semantic model loaded.")

# ---------- 4. Analyze email ----------
def analyze_email(email):
    email_text = "" if email is None else str(email).strip()
    if email_text == "":
        return "⚠️ No text provided.", "0%", "Please paste the email content or URL."

    explanation_lines = []

    # Extract URLs and domains
    urls = extract_urls(email_text)
    reg_domains = [get_registered_domain(domain_from_url(u).replace("www.", "")) for u in urls]

    safe_hits = [d for d in reg_domains if d in SAFE_DOMAINS]
    non_safe_hits = [d for d in reg_domains if d not in SAFE_DOMAINS]

    if reg_domains:
        explanation_lines.append("🔗 URLs found: " + ", ".join(urls))
        if safe_hits and not non_safe_hits:
            explanation_lines.append(f"✅ All URL domains are whitelisted: {', '.join(safe_hits)}")
            return "✅ Safe Email", "99% 💪", "\n".join(explanation_lines)
        else:
            if non_safe_hits:
                explanation_lines.append("⚠️ Non-whitelisted domains detected: " + ", ".join(non_safe_hits))
            if safe_hits:
                explanation_lines.append("✅ Also found whitelisted domains: " + ", ".join(safe_hits))
    else:
        explanation_lines.append("📄 No URLs detected.")

    # ---------- Semantic analysis ----------
    try:
        email_embedding = semantic_model.encode(email_text, convert_to_tensor=True)
        if urls:
            # Check similarity of email text to URLs (heuristic)
            url_embeddings = semantic_model.encode(urls, convert_to_tensor=True)
            cosine_scores = util.cos_sim(email_embedding, url_embeddings)
            max_score = float(cosine_scores.max())
        else:
            max_score = 0
    except Exception as e:
        explanation_lines.append(f"⚠️ Embedding error: {e}")
        max_score = 0

    # Keyword hits
    hits = [k for k in SUSPICIOUS_KEYWORDS if k in email_text.lower()]
    if hits:
        explanation_lines.append("⚠️ Suspicious keywords detected: " + ", ".join(hits))

    # Decision threshold
    if max_score > 0.6 or hits:
        label = "🚨 Phishing Email"
        confidence = f"{int(max(max_score, 0.7) * 100)}% 💪"
    else:
        label = "✅ Legitimate Email"
        confidence = f"{int((1 - max_score) * 100)}% 🙂"

    explanation_lines.append(f"Semantic similarity score: {max_score:.3f}")
    return label, confidence, "\n".join(explanation_lines)

# ---------- 5. Gradio UI ----------
with gr.Blocks(title="PHISHING EMAIL DETECTOR") as demo:
    gr.Markdown("## 🕵️‍♂️ PHISHING EMAIL DETECTOR")

    with gr.Row():
        inp = gr.Textbox(
            lines=8,
            placeholder="Paste the email content or URL here...",
            label="Email / Message"
        )

    with gr.Row():
        btn = gr.Button("Analyze")

    with gr.Row():
        out_label = gr.Textbox(label="Result")
        out_conf = gr.Textbox(label="Confidence (%)")

    out_expl = gr.Textbox(label="Details", lines=8)

    btn.click(fn=analyze_email, inputs=[inp], outputs=[out_label, out_conf, out_expl])

# ---------- 6. Launch App ----------
if __name__ == "__main__":
    print("\n🌐 Launching Gradio app...")
    demo.launch(share=True)
