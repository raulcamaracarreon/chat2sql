# nlp2sql.py
import os
import httpx
from typing import Optional
from openai import OpenAI

class NLtoSQL:
    """
    Traductor NL→SQL con dos proveedores:
      - provider='openai'  -> usa OpenAI Chat Completions
      - provider='ollama'  -> usa servidor local Ollama en /api/chat
    """
    def __init__(
        self,
        system_prompt: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.system_prompt = system_prompt
        self.provider = provider.lower().strip()
        self.model = model
        self.ollama_base_url = ollama_base_url.rstrip("/")

        if self.provider == "openai":
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Falta OPENAI_API_KEY (en .env o en el formulario).")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "ollama":
            # Cliente HTTP para Ollama local
            self.http = httpx.Client(base_url=self.ollama_base_url, timeout=60.0)
        else:
            raise ValueError("Proveedor no soportado. Usa 'openai' u 'ollama'.")

    def nl_to_sql(self, user_query: str) -> str:
        if self.provider == "openai":
            rsp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            sql = rsp.choices[0].message.content.strip()

        elif self.provider == "ollama":
            # API de chat de Ollama: POST /api/chat
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                ],
            }
            r = self.http.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            if "message" in data and "content" in data["message"]:
                sql = data["message"]["content"].strip()
            elif "response" in data:  # por si usamos /api/generate en algún momento
                sql = data["response"].strip()
            else:
                raise RuntimeError(f"Respuesta de Ollama inesperada: {data!r}")

        else:
            raise RuntimeError("Proveedor no soportado.")

        # Limpieza de fences si el modelo los devuelve
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
