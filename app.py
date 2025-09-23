from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from db import CSVDb
from nlp2sql import NLtoSQL
from prompts import build_system_prompt
from sql_guard import is_safe_select
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

STATE = {
    "db": None,
    "table": None,
    "schema_text": None,
    "model": None,
    "provider": "openai",
    "openai_model": "gpt-4o-mini",
    "ollama_model": "llama3.1:8b-instruct",
    "ollama_base_url": "http://localhost:11434",
}

DEFAULT_LIMIT = 1000

def apply_default_limit(sql: str, limit: int = DEFAULT_LIMIT) -> str:
    low = sql.lower()
    if " limit " in low:
        return sql
    return f"SELECT * FROM ({sql}) AS _sub LIMIT {limit}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1) Cargar CSV + configurar proveedor/modelo
        if "csv_file" in request.files:
            f = request.files["csv_file"]
            if not f.filename or not f.filename.lower().endswith(".csv"):
                flash("Sube un archivo .csv", "error"); return redirect(url_for("index"))

            df = pd.read_csv(f)
            table_name = (request.form.get("table_name") or "data").strip() or "data"

            provider = (request.form.get("provider") or STATE["provider"]).lower().strip()

            # OpenAI
            openai_model = request.form.get("openai_model", STATE["openai_model"])
            openai_api_key = request.form.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")

            # Ollama
            ollama_model = request.form.get("ollama_model", STATE["ollama_model"]).strip() or STATE["ollama_model"]
            ollama_base_url = (request.form.get("ollama_base_url") or STATE["ollama_base_url"]).strip()

            db = CSVDb()
            db.load_dataframe(df, table_name)

            schema_text = db.describe_schema(table_name)
            system_prompt = build_system_prompt(schema_text, dialect="SQLite")

            if provider == "openai":
                nltosql = NLtoSQL(
                    system_prompt=system_prompt,
                    provider="openai",
                    model=openai_model,
                    openai_api_key=openai_api_key,
                )
            else:
                nltosql = NLtoSQL(
                    system_prompt=system_prompt,
                    provider="ollama",
                    model=ollama_model,
                    ollama_base_url=ollama_base_url,
                )

            STATE.update({
                "db": db,
                "table": table_name,
                "schema_text": schema_text,
                "model": nltosql,
                "provider": provider,
                "openai_model": openai_model,
                "ollama_model": ollama_model,
                "ollama_base_url": ollama_base_url,
            })

            flash(f"CSV cargado en tabla '{table_name}' con {len(df)} filas.", "success")
            return redirect(url_for("index"))

        # 2) NL→SQL + ejecutar
        elif "user_query" in request.form:
            if not (STATE["db"] and STATE["model"] and STATE["table"]):
                flash("Primero carga un CSV.", "error"); return redirect(url_for("index"))

            user_query = request.form["user_query"].strip()
            if not user_query:
                flash("Escribe una pregunta.", "error"); return redirect(url_for("index"))

            try:
                sql = STATE["model"].nl_to_sql(user_query)
                sql = sql.replace("```sql", "").replace("```", "").strip()

                ok, reason, clean = is_safe_select(sql)
                if not ok:
                    flash(f"Consulta bloqueada: {reason}", "error"); return redirect(url_for("index"))

                final_sql = apply_default_limit(clean)
                df = STATE["db"].query(final_sql)
                preview = df.head(DEFAULT_LIMIT)
                return render_template(
                    "index.html",
                    table_name=STATE["table"],
                    schema_text=STATE["schema_text"],
                    last_sql=final_sql,
                    result_html=preview.to_html(classes="table table-sm table-striped", index=False, border=0),
                    provider=STATE["provider"],
                    openai_model=STATE["openai_model"],
                    ollama_model=STATE["ollama_model"],
                    ollama_base_url=STATE["ollama_base_url"],
                )
            except Exception as e:
                flash(f"Error: {e}", "error"); return redirect(url_for("index"))

    # GET inicial o tras redirect
    return render_template(
        "index.html",
        table_name=STATE["table"],
        schema_text=STATE["schema_text"],
        last_sql=None,
        result_html=None,
        provider=STATE["provider"],
        openai_model=STATE["openai_model"],
        ollama_model=STATE["ollama_model"],
        ollama_base_url=STATE["ollama_base_url"],
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
