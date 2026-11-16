"""OllamaLLM contra un servidor HTTP falso que imita /api/chat (sin red ni modelo real)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from paisajia.agent import explain
from paisajia.agent.llm import OUTPUT_SCHEMA, OllamaLLM, get_llm

LOG = [
    {"id": "R1-1", "ronda": 1, "eje": "hora", "variante": "atardecer", "decision": "aceptada", "motivo": "me encanta la luz"},
    {"id": "R1-2", "ronda": 1, "eje": "clima", "variante": "niebla", "decision": "rechazada", "motivo": None},
]
GOOD = {"aceptadas": ["R1-1"], "rechazadas": ["R1-2"],
        "resumen": "El usuario aceptó atardecer (R1-1). Rechazó niebla (R1-2)."}
BAD = {"aceptadas": [], "rechazadas": ["R1-1", "R1-2"], "resumen": "Rechazó todo."}


def _server(responses, status=200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if status != 200:
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b'{"error": "model not found"}')
                return
            content = json.dumps(responses[min(len(calls) - 1, len(responses) - 1)], ensure_ascii=False)
            out = json.dumps({"message": {"role": "assistant", "content": content}, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


@pytest.fixture
def make_llm():
    servers = []

    def _make(responses, status=200):
        srv, calls = _server(responses, status)
        servers.append(srv)
        return OllamaLLM(model="qwen2.5:3b", host=f"127.0.0.1:{srv.server_port}"), calls

    yield _make
    for s in servers:
        s.shutdown()


def test_valid_first_try(make_llm):
    llm, calls = make_llm([GOOD])
    exp = explain(LOG, {}, llm)
    assert exp.first_try_valid and not exp.used_template
    assert calls[0]["format"] == OUTPUT_SCHEMA and calls[0]["model"] == "qwen2.5:3b"
    assert calls[0]["messages"][0]["role"] == "system"


def test_retry_sends_verifier_feedback(make_llm):
    llm, calls = make_llm([BAD, GOOD])
    exp = explain(LOG, {}, llm)
    assert exp.attempts == 2 and not exp.used_template
    assert "verificador" in calls[1]["messages"][-1]["content"]


def test_missing_model_falls_back_to_template(make_llm):
    llm, _ = make_llm([], status=404)
    exp = explain(LOG, {}, llm)
    assert exp.used_template
    assert any("ollama pull" in e for attempt in exp.errors for e in attempt)


def test_get_llm_aliases():
    assert isinstance(get_llm("ollama"), OllamaLLM)
    assert get_llm("qwen").model == "qwen2.5:3b"
