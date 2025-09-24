from flask import Flask, render_template, request, redirect, url_for, flash, session
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

# Cookies de sesión un poco más seguras en prod (Render)
IS_PROD = os.environ.get("RENDER", "").lower() in {"true", "1"} or os.environ.get("FLASK_ENV") == "production"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PROD,  # en HTTPS se marca Secure
)

STATE = {"db": None, "table": None, "schema_text": None, "model": None}
DEFAULT_LIMIT = 1000

# --------- Auth ----------
def is_authenticated() -> bool:
    return bool(session.get("auth"))

@app.before_request
def require_login():
    # Permite login, logout, salud y estáticos
    if request.path.startswith("/static/"):
        return
    if request.endpoint in {"login", "logout", "health"}:
        return
    # Bloquea todo lo demás si no hay sesión
    if not is_authenticated():
        return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    configured_password = os.environ.get("APP_PASSWORD")
    if not configured_password:
        return "APP_PASSWORD no está configurada en el entorno.", 500

    if request.method == "POST":
        pwd = request.form.get("password", "")
        if pwd == configured_password:
            session["auth"] = True
            flash("Sesión iniciada.", "success")
            return redirect(url_for("index"))
        else:
            flash("Contraseña incorrecta.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Sesión cerrada.", "success")
    return redirect(url_for("login"))

@app.route("/health")
def health():
    return {"ok": True}

# --------- App principal ----------
def apply_default_limit(sql: str, limit: int = DEFAULT_LIMIT) -> str:
    low = sql.lower()
    if " limit " in low:
        return sql
    return f"SELECT * FROM ({sql}) AS _sub LIMIT {limit}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1) Cargar CSV
        if "csv_file" in request.files:
            f = request.files["csv_file"]
            if not f.filename or not f.filename.lower().endswith(".csv"):
                flash("Sube un archivo .csv", "error")
                return redirect(url_for("index"))

            df = pd.read_csv(f)
            table_name = (request.form.get("table_name") or "data").strip() or "data"

            db = CSVDb()
            db.load_dataframe(df, table_name)

            schema_text = db.describe_schema(table_name)
            system_prompt = build_system_prompt(schema_text, dialect="SQLite")

            # Modelo fijo por defecto: gpt-4o-mini (API desde ENV)
            nltosql = NLtoSQL(system_prompt=system_prompt, model="gpt-4o-mini")

            STATE.update({"db": db, "table": table_name, "schema_text": schema_text, "model": nltosql})
            flash(f"CSV cargado en tabla '{table_name}' con {len(df)} filas.", "success")
            return redirect(url_for("index"))

        # 2) NL→SQL + ejecutar (solo lectura)
        elif "user_query" in request.form:
            if not (STATE["db"] and STATE["model"] and STATE["table"]):
                flash("Primero carga un CSV.", "error")
                return redirect(url_for("index"))

            user_query = request.form["user_query"].strip()
            if not user_query:
                flash("Escribe una pregunta.", "error")
                return redirect(url_for("index"))

            try:
                sql = STATE["model"].nl_to_sql(user_query).strip()
                ok, reason, clean = is_safe_select(sql)
                if not ok:
                    flash(f"Consulta bloqueada: {reason}", "error")
                    return redirect(url_for("index"))

                final_sql = apply_default_limit(clean)
                df = STATE["db"].query(final_sql)
                preview = df.head(DEFAULT_LIMIT)
                return render_template(
                    "index.html",
                    table_name=STATE["table"],
                    schema_text=STATE["schema_text"],
                    last_sql=final_sql,
                    result_html=preview.to_html(classes="table table-sm table-striped", index=False, border=0),
                )
            except Exception as e:
                flash(f"Error: {e}", "error")
                return redirect(url_for("index"))

    # GET inicial
    return render_template(
        "index.html",
        table_name=STATE["table"],
        schema_text=STATE["schema_text"],
        last_sql=None,
        result_html=None,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

