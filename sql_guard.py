# sql_guard.py
import re

_FORBIDDEN_WORDS = re.compile(
    r"\b(attach|detach|vacuum|pragma|drop|create|alter|update|insert|delete|replace|"
    r"truncate|grant|revoke|rename|transaction|begin|commit|rollback|savepoint|release|"
    r"analyze|reindex)\b",
    flags=re.IGNORECASE,
)

def _strip_sql_comments(sql: str) -> str:
    # quita -- ... y /* ... */ para evitar ocultar palabras peligrosas
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql

def is_safe_select(sql: str) -> tuple[bool, str, str]:
    """
    Verifica que sea 1 sola sentencia SELECT/WITH y sin palabras peligrosas.
    Devuelve (ok, motivo, sql_limpio_sin_puntoycoma_final).
    """
    if not sql or not sql.strip():
        return False, "Consulta vacía.", sql

    s = _strip_sql_comments(sql).strip()

    # No múltiples sentencias con ';' (permitimos ';' final)
    s_no_last = s[:-1] if s.endswith(";") else s
    if ";" in s_no_last:
        return False, "No se permiten múltiples sentencias.", s

    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False, "Solo se permiten consultas SELECT (o WITH ... SELECT).", s

    if _FORBIDDEN_WORDS.search(low):
        return False, "Se detectó una palabra clave no permitida para solo-lectura.", s

    # normaliza: quita ';' final si lo hay
    s = s.rstrip().rstrip(";").rstrip()
    return True, "", s
