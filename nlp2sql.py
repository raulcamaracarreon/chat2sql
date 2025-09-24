# nlp2sql.py
import os
from typing import Optional
from openai import OpenAI

class NLtoSQL:
    """
    Traductor NL→SQL con OpenAI.
    Modelo por defecto: gpt-4o-mini.
    """
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini", openai_api_key: Optional[str] = None):
        self.system_prompt = system_prompt
        self.model = model
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta OPENAI_API_KEY en variables de entorno.")
        self.client = OpenAI(api_key=api_key)

    def nl_to_sql(self, user_query: str) -> str:
        rsp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        sql = rsp.choices[0].message.content.strip()
        # Limpia fences si vinieran
        return sql.replace("```sql", "").replace("```", "").strip()

