import gradio as gr

from phishguard.__main__ import score_url

EXAMPLES = [
    "https://example.com",
    "http://192.168.0.14/secure-login/verify-account",
    "https://update-confirm.example.net/wallet/invoice",
]


def score(url):
    proba, threshold, top, _ = score_url(url)
    label = "phishing" if proba >= threshold else "likely legit"
    return proba, label, top, threshold


def predict(url):
    url = (url or "").strip()
    if not url:
        return "", "", ""
    proba, label, top, threshold = score(url)
    feats = "\n".join(f"{name:14}  {c:+.3f}   raw={raw:g}" for name, c, raw in top)
    return f"{proba:.3f}", f"{label}  (threshold {threshold:.2f})", feats


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="URL", placeholder="https://example.com"),
    outputs=[
        gr.Textbox(label="phishing probability"),
        gr.Textbox(label="label"),
        gr.Textbox(label="top features", lines=6),
    ],
    examples=EXAMPLES,
    title="PhishGuard",
    description="Lexical URL score from the saved NumPy weights. Course project, not a production filter.",
)

if __name__ == "__main__":
    demo.launch()
