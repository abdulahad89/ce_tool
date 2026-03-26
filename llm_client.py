from __future__ import annotations

from huggingface_hub import InferenceClient

from config import HF_MODEL_ID, HF_TOKEN


def get_hf_client() -> InferenceClient:
    """Create a Hugging Face InferenceClient, with token if provided."""
    if HF_TOKEN:
        return InferenceClient(api_key=HF_TOKEN)
    return InferenceClient()


def call_deepseek_r1(query: str, context: str, max_tokens: int = 512) -> str:
    """Call DeepSeek-R1 (or compatible) via chat.completions API."""
    client = get_hf_client()

    system_prompt = (
        "You are an analytics assistant for marketing campaigns. "
        "You are given structured context with conversion and channel engagement metrics. "
        "Use ONLY the provided context to answer, be concise and data-driven, and clearly state "
        "which campaigns, treatments, or channels you are referring to."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Here is the campaign data context (conversion + engagement):\n\n{context}\n\n"
        "Now answer the question step by step, then give a short final conclusion."
    )

    try:
        completion = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        message = completion.choices[0].message
        if isinstance(message, dict):
            return message.get("content", "(No content returned)")
        return str(message)

    except Exception as e:
        return f"DeepSeek-R1 call failed: {e}"
