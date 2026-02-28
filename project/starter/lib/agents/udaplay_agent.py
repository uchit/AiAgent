from lib.rag import GameRAG
from lib.parsers import evaluate_confidence
from lib.tooling import GameWebSearch


class UdaPlayAgent:
    def __init__(self, vector_db):
        self.rag = GameRAG(vector_db)
        self.web = GameWebSearch()

    def run(self, question: str) -> dict:
        context = self.rag.retrieve(question)
        confidence = evaluate_confidence(context)

        source = "internal"

        if confidence < 0.7:
            context = self.web.search(question)
            confidence = 0.85
            source = "web"

        answer = self.rag.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "source": source
        }