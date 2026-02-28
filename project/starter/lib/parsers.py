from lib.llm import chat_completion


def evaluate_confidence(context: str) -> float:
    prompt = f"""
    Rate how confidently the following context answers the question.
    Return ONLY a number between 0 and 1.

    Context:
    {context}
    """

    score = chat_completion(prompt)
    return float(score.strip())