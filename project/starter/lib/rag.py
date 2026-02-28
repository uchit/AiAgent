from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GameRAG:
    def retrieve(self, vector_db, question: str):
        docs = vector_db.search(question)
        return "\n".join(docs)

    def generate(self, question: str, context: str):
        prompt = f"""
        Answer the question using the context.

        Question:
        {question}

        Context:
        {context}
        """
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content