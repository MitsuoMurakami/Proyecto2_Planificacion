"""Interfaz Gradio de PaisajIA.

Ejecutar:  python app.py            (modo según variables PAISAJIA_*; por defecto sin GPU)
           python app.py --generator flux --share
"""
from __future__ import annotations

import argparse
import html
import logging
import os
import re

import gradio as gr

from paisajia import metrics
from paisajia.config import AI_LABEL, AXES, AXIS_NAMES, N_VARIATIONS, SETTINGS
from paisajia.guardrails import check_instruction
from paisajia.registry import Registry
from paisajia.session import StudioSession

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
REGISTRY = Registry()
N = N_VARIATIONS

MODE_ALL = "Muéstrame una idea de cada tipo"
MODE_PICK = "Quiero elegir qué cambiar"

HEADER = f"""
<header class="pj-header">
  <div class="pj-brand">PaisajIA</div>
  <p class="pj-tagline">Mira tu paisaje en otra luz, otra estación o con otro estilo. Elige las versiones que te gustan
  y cada ronda nueva se ajusta a tus elecciones.</p>
  <p class="pj-note">Usa una foto tuya o con permiso, sin personas. Las imágenes creadas llevan la marca «{AI_LABEL}».</p>
</header>
"""


def _step(n: int, title: str, hint: str = "") -> str:
    hint_html = f'<span class="pj-step-hint">{hint}</span>' if hint else ""
    return f'<div class="pj-step"><span class="pj-step-n">{n}</span><span class="pj-step-t">{title}</span>{hint_html}</div>'


def _esc(s: str | None) -> str:
    return html.escape(s or "")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _friendly_reason(plan_reason: str | None) -> str:
    """Traduce el motivo del planificador a lenguaje de usuario."""
    r = (plan_reason or "").split(":", 1)[-1].strip()
    if r.startswith("elegida por el usuario"):
        return "La elegiste tú"
    if r.startswith("variante base"):
        return "Una primera idea para empezar"
    if r.startswith("variante adicional"):
        return "Otra opción del mismo tipo"
    if r.startswith("aceptada"):
        return "Te gustó: aquí tienes otra versión"
    if r.startswith("alternativa a"):
        prev = r.removeprefix("alternativa a").removesuffix(", rechazada").strip()
        return f"En lugar de «{prev}», que no te gustó"
    if "instrucción propia" in r:
        return "Tu idea, en otra versión"
    return r[:1].upper() + r[1:]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _card_caption(v) -> str:
    state = {"aceptada": ("liked", "Te gusta"), "rechazada": ("disliked", "No te gusta")}.get(
        v.decision or "", ("pending", "Sin elegir"))
    why = _friendly_reason(v.plan_reason)
    warn = f'<div class="pj-card-warn">{_esc(v.warning)}</div>' if v.warning else ""
    title = v.variant if v.axis in AXES else (v.custom_text or v.variant)
    return (f'<div class="pj-card-cap">'
            f'<div class="pj-card-top"><span class="pj-card-axis">{_esc(v.axis_label)}</span>'
            f'<span class="pj-chip pj-chip-{state[0]}">{state[1]}</span></div>'
            f'<div class="pj-card-title">{_esc(_cap(title))}</div>'
            f'<div class="pj-card-why">{_esc(why)}</div>{warn}</div>')


def _slot_outputs(sess: StudioSession | None):
    """Imagen, leyenda y motivo para cada una de las cinco posiciones."""
    out = []
    cur = sess.current() if sess else []
    for i in range(N):
        if i < len(cur):
            v = cur[i]
            out += [v.labeled_path, _card_caption(v), gr.update(value=v.reason or "")]
        else:
            out += [None, "", gr.update(value="")]
    return out


def _status(sess: StudioSession | None, msg: str = "") -> str:
    if not sess or not sess.sid:
        return f'<div class="pj-status">{msg}</div>' if msg else ""
    cur = sess.current()
    done = sum(1 for v in cur if v.decision)
    left = len(cur) - done
    progress = (f"Te faltan {left} por elegir." if left else
                "Listo. Ya puedes ver el resumen de tus elecciones.")
    body = f"<b>Ronda {sess.round}</b> · {done} de {len(cur)} elegidas. {progress}"
    if msg:
        body += f'<div class="pj-status-msg">{msg}</div>'
    return f'<div class="pj-status">{body}</div>'


def _tech(sess: StudioSession | None) -> str:
    if not sess or not sess.sid:
        return ""
    t, lp = sess.round_seconds.get(sess.round), sess.lpips.get(sess.round)
    parts = [f"Generador: `{sess.generator.name}`", f"Agente: `{sess.llm.name}`"]
    if t is not None:
        parts.append(f"Generación de la ronda: {t:.1f} s")
    if lp is not None:
        parts.append(f"Diversidad (LPIPS): {lp:.2f}")
    sims = [f"{v.var_id} {v.dino:.2f}" for v in sess.current() if v.dino is not None]
    if sims:
        parts.append("Similitud con la foto original (DINOv2): " + ", ".join(sims))
    return "  \n".join(parts)


def _refresh(sess, msg=""):
    can_summarize = bool(sess and sess.all_decided() and sess.round not in sess.explanations)
    return [sess, _status(sess, msg), gr.update(interactive=can_summarize), _tech(sess)] + _slot_outputs(sess)


def _fail(msg: str):
    return _refresh(None, msg) + [gr.update(), gr.update(), gr.update(interactive=False)]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def on_start(image, mode, axes, custom_text, *variants):
    if image is None:
        return _fail("Primero sube una foto de paisaje.")
    custom_text = (custom_text or "").strip() or None
    if custom_text:
        g = check_instruction(custom_text)
        if not g.allowed:
            REGISTRY.log_guardrail(None, "instruccion", g.category, custom_text)
            return _fail(_esc(g.message))
    sess = StudioSession(REGISTRY)
    pre = sess.start(image)
    if not pre.ok:
        return _fail("No podemos usar esta foto: " + _esc("; ".join(pre.reasons)))
    if mode == MODE_ALL:
        sel_axes = None if not custom_text else list(AXIS_NAMES)[: N - 1]
        chosen = {}
    else:
        sel_axes = [a for a in AXIS_NAMES if AXES[a]["label"] in (axes or [])]
        if not sel_axes and not custom_text:
            return _fail("Elige al menos un tipo de cambio o describe tu idea.")
        chosen = {a: variants[AXIS_NAMES.index(a)] for a in sel_axes}
    try:
        sess.first_round(sel_axes, chosen, custom_text)
    except PermissionError as exc:
        return _fail(_esc(str(exc)))
    return _refresh(sess, "Estas son tus cinco versiones. Marca las que te gustan.") + [
        "", None, gr.update(interactive=False)]


def on_decide(sess, idx, decision, reason):
    if not sess or idx >= len(sess.current()):
        return _refresh(sess, "Todavía no hay una imagen en ese lugar.")
    if sess.round in sess.explanations:
        return _refresh(sess, "Esta ronda ya está cerrada. Pasa a la siguiente.")
    v = sess.current()[idx]
    sess.decide(v.var_id, decision, reason)
    return _refresh(sess)


def on_regenerate(sess, idx):
    if not sess or idx >= len(sess.current()):
        return _refresh(sess, "Todavía no hay una imagen en ese lugar.")
    if sess.round in sess.explanations:
        return _refresh(sess, "Esta ronda ya está cerrada.")
    v = sess.regenerate(sess.current()[idx].var_id)
    return _refresh(sess, f"Nueva versión de «{_esc(v.variant)}» lista.")


def on_bulk(sess, decision, comment):
    if not sess or not sess.sid:
        return _refresh(sess, "Primero sube una foto y crea tus versiones.")
    if sess.round in sess.explanations:
        return _refresh(sess, "Esta ronda ya está cerrada.")
    n = sess.decide_pending(decision, comment)
    return _refresh(sess, "" if n else "No quedaban imágenes sin elegir.")


def _plan_rows(table):
    return [[r.posicion, r.anterior, r.decision, r.accion, r.siguiente, r.motivo,
             ", ".join(f"{k}={v}" for k, v in r.probabilidades.items())] for r in table]


def _names(sess, ids):
    by_id = {v.var_id: v for v in sess.current()}
    return [f"{_cap(by_id[i].variant)} ({by_id[i].axis_label.lower()})" if i in by_id else i for i in ids]


def on_summarize(sess):
    if not sess or not sess.all_decided():
        return ["Elige todas las imágenes para ver el resumen.", None, [], gr.update(interactive=False)]
    exp, table = sess.summarize()
    o = exp.output
    liked = _names(sess, o["aceptadas"])
    disliked = _names(sess, o["rechazadas"])
    li = "".join(f"<li>{_esc(x)}</li>" for x in liked) or "<li class='pj-empty'>Ninguna</li>"
    di = "".join(f"<li>{_esc(x)}</li>" for x in disliked) or "<li class='pj-empty'>Ninguna</li>"
    nxt = "".join(f"<li>{_esc(_cap(r.siguiente))}</li>" for r in table)
    # El agente cita IDs (R1-2) para que el verificador los compruebe; al usuario se le muestra el texto sin ellos.
    text = re.sub(r"\s*\((?:R\d+-\d+(?:,\s*)?)+\)", "", o["resumen"])
    md = (f'<div class="pj-summary">'
          f'<h3>Lo que aprendimos de ti en la ronda {sess.round}</h3>'
          f'<p class="pj-summary-text">{_esc(text)}</p>'
          f'<div class="pj-summary-cols">'
          f'<div><h4>Te gustó</h4><ul>{li}</ul></div>'
          f'<div><h4>No te gustó</h4><ul>{di}</ul></div>'
          f'<div><h4>En la próxima ronda verás</h4><ul>{nxt}</ul></div>'
          f'</div></div>')
    badge = ("verificado al primer intento" if exp.first_try_valid else
             f"verificado tras {exp.attempts} intentos" if not exp.used_template else
             "plantilla determinista (el LLM no pasó la verificación)")
    debug = {"fuente": exp.source, "verificacion": badge, "segundos": round(exp.seconds, 1),
             "salida": o, "errores_por_intento": exp.errors}
    return [md, debug, _plan_rows(table), gr.update(interactive=True)]


def on_next(sess):
    if not sess or sess.plan is None:
        return _refresh(sess, "Primero mira el resumen de esta ronda.") + [gr.update(interactive=False)]
    sess.next_round()
    return _refresh(sess, "Nueva ronda preparada según lo que elegiste.") + [gr.update(interactive=False)]


def on_close(sess):
    if sess:
        sess.close(delete_files=True)
    return _refresh(None, "Sesión terminada. Borramos las imágenes creadas; no guardamos tu foto.") + [
        "", None, [], gr.update(interactive=False)]


def on_monitor(include_sim):
    m = metrics.summary(REGISTRY, include_simulated=include_sim)
    f = lambda x, p=False: "—" if x is None else (f"{x:.0%}" if p else f"{x:.2f}")  # noqa: E731
    c, d, t, it, ag, us = m["composicion"], m["diversidad"], m["tiempo"], m["iteracion"], m["agente"], m["usuarios"]
    md = f"""
| Dimensión | Métrica | Valor | Meta |
|---|---|---|---|
| Uso | sesiones / rondas / decisiones | {m['sesiones']} / {m['rondas']} / {m['decisiones']} | — |
| Composición | similitud media ({', '.join(c['backend']) or '—'}) | {f(c['similitud_media'])} | — |
| Composición | % variaciones sobre el umbral | {f(c['pct_sobre_umbral'], True)} | {c['meta']} |
| Diversidad | LPIPS medio por ronda ({', '.join(d['backend']) or '—'}) | {f(d['lpips_medio'])} | {d['meta']} |
| Tiempo | segundos por ronda (máx.) | {f(t['segundos_por_ronda'])} ({f(t['max'])}) | {t['meta']} |
| Iteración | aceptación ronda 1 → rondas ≥ 2 | {f(it['aceptacion_ronda1'], True)} → {f(it['aceptacion_rondas_2+'], True)} | mayor en rondas planificadas |
| Agente | válido al primer intento | {f(ag['validos_al_primer_intento'], True)} | — |
| Agente | uso de plantilla de respaldo | {f(ag['uso_de_plantilla'], True)} | — |
| Agente | coincidencia final con el registro | {f(ag['coincidencia_final_con_registro'], True)} | 100 % |
| Usuarios | SUS medio ({us['encuestas']} encuestas) | {f(us['sus_medio'])} | ≥ 70 |
| Usuarios | claridad / fidelidad (1-5) | {f(us['claridad_media'])} / {f(us['fidelidad_media'])} | ≥ 4 |
| Usuarios | tiempo medio por decisión (s) | {f(us['tiempo_medio_por_decision_s'])} | — |
"""
    axes = [[k, v["n"], f(v["aceptación"], True)] for k, v in m["aceptacion_por_eje"].items()]
    guard = [[g["kind"], g["category"], g["n"]] for g in m["guardrails"]]
    return md, axes, guard, m["agente"]["errores_frecuentes"]


def on_survey(sess, comment, clarity, fidelity, *answers):
    try:
        score = metrics.sus_score([int(a) for a in answers])
    except (TypeError, ValueError):
        return "Responde las 10 preguntas para enviar tu opinión."
    REGISTRY.save_survey(sess.sid if sess else None, [int(a) for a in answers], score, int(clarity), int(fidelity), comment or "")
    return "¡Gracias por tu opinión! Nos ayuda a mejorar PaisajIA."


# ---------------------------------------------------------------------------
# Diseño visual: fondo blanco, una sola tinta de acento y jerarquía por pasos
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#eef4f1", c100="#d6e6de", c200="#b0cdbf", c300="#85b09c", c400="#5c917a",
                                c500="#3f765f", c600="#2f5d4a", c700="#264c3d", c800="#1e3c30", c900="#162d24",
                                c950="#0e1d17"),
    neutral_hue="stone",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f7f6f3",
    block_background_fill="#ffffff",
    block_border_color="#e7e5e0",
    block_border_width="1px",
    block_shadow="none",
    block_label_background_fill="#ffffff",
    block_label_text_color="#57534e",
    block_title_text_color="#292524",
    body_text_color="#1c1917",
    body_text_color_subdued="#78716c",
    input_background_fill="#ffffff",
    input_border_color="#d6d3cd",
    button_primary_background_fill="#2f5d4a",
    button_primary_background_fill_hover="#264c3d",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f5f5f2",
    button_secondary_border_color="#d6d3cd",
    button_secondary_text_color="#292524",
    button_border_width="1px",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_selected="#eef4f1",
    checkbox_label_border_color="#d6d3cd",
    checkbox_label_border_color_selected="#2f5d4a",
    checkbox_background_color_selected="#2f5d4a",
)

CSS = """
:root { color-scheme: light; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; background: #fff !important; }
footer { display: none !important; }

.pj-header { padding: 28px 0 8px; border-bottom: 1px solid #e7e5e0; margin-bottom: 8px; }
.pj-brand { font-family: 'Fraunces', Georgia, serif; font-size: 34px; font-weight: 600; letter-spacing: -0.02em;
            color: #1c1917; }
.pj-tagline { font-size: 17px; color: #44403c; max-width: 680px; margin: 6px 0 4px; line-height: 1.5; }
.pj-note { font-size: 13.5px; color: #78716c; margin: 0 0 12px; }

.pj-step { display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 10px; row-gap: 2px; margin: 22px 0 6px; }
.pj-step-n { width: 26px; height: 26px; border-radius: 50%; background: #2f5d4a; color: #fff; font-size: 14px;
             font-weight: 600; display: inline-flex; align-items: center; justify-content: center; flex: none; }
.pj-step-t { font-family: 'Fraunces', Georgia, serif; font-size: 20px; font-weight: 600; color: #1c1917;
             white-space: nowrap; }
.pj-step-hint { font-size: 14px; color: #78716c; }

.pj-status { background: #f7f6f3; border-left: 3px solid #2f5d4a; padding: 10px 14px; border-radius: 4px;
             color: #292524; font-size: 15px; }
.pj-status-msg { color: #57534e; margin-top: 2px; }

.pj-card { border: 1px solid #e7e5e0 !important; border-radius: 8px !important; padding: 8px !important;
           background: #fff !important; gap: 8px !important; }
.pj-card-cap { padding: 2px 2px 0; }
.pj-card-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
.pj-card-axis { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; color: #78716c; }
.pj-card-title { font-size: 16px; font-weight: 600; color: #1c1917; margin-top: 2px; }
.pj-card-why { font-size: 13px; color: #78716c; margin-top: 1px; }
.pj-card-warn { font-size: 12.5px; color: #92400e; background: #fffbeb; padding: 4px 6px; border-radius: 4px;
                margin-top: 6px; }
.pj-chip { font-size: 12px; padding: 1px 8px; border-radius: 999px; white-space: nowrap; }
.pj-chip-pending { background: #f5f5f4; color: #78716c; }
.pj-chip-liked { background: #dcfce7; color: #166534; }
.pj-chip-disliked { background: #fee2e2; color: #991b1b; }
.pj-card .row { flex-wrap: nowrap !important; gap: 6px !important; }
.pj-card .row > button { min-width: 0 !important; flex: 1 1 0 !important; border: 1px solid #d6d3cd !important;
                         padding: 6px 4px !important; font-size: 13.5px !important; }
.pj-card .row > button.pj-like { border-color: #2f5d4a !important; color: #2f5d4a !important; font-weight: 600; }
.pj-card .row > button:hover { background: #f5f5f2 !important; }
.pj-quiet { border: none !important; background: transparent !important; color: #57534e !important;
            text-decoration: underline; text-underline-offset: 3px; box-shadow: none !important; }

.pj-summary { border: 1px solid #e7e5e0; border-radius: 8px; padding: 18px 20px; background: #fff; }
.pj-summary h3 { font-family: 'Fraunces', Georgia, serif; font-size: 20px; margin: 0 0 8px; color: #1c1917; }
.pj-summary-text { color: #292524; line-height: 1.6; margin: 0 0 14px; }
.pj-summary-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
.pj-summary h4 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: #78716c; margin: 0 0 6px; }
.pj-summary ul { margin: 0; padding-left: 18px; color: #292524; }
.pj-summary .pj-empty { color: #a8a29e; list-style: none; margin-left: -18px; }

.pj-panel-note { font-size: 14px; color: #78716c; }
"""

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap" '
        'rel="stylesheet">')

# Fuerza el modo claro aunque el sistema del usuario esté en modo oscuro.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="PaisajIA") as demo:
        state = gr.State(None)
        gr.HTML(HEADER)
        with gr.Tabs():
            with gr.Tab("Estudio"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.HTML(_step(1, "Sube tu foto", "Un paisaje: montaña, costa, bosque, desierto o nieve"))
                        inp = gr.Image(type="pil", show_label=False, height=340, sources=["upload", "clipboard"])
                    with gr.Column(scale=6):
                        gr.HTML(_step(2, "Elige qué quieres ver"))
                        mode = gr.Radio([MODE_ALL, MODE_PICK], value=MODE_ALL, show_label=False)
                        with gr.Group(visible=False) as pick_box:
                            axes = gr.CheckboxGroup([AXES[a]["label"] for a in AXIS_NAMES], label="¿Qué cambiamos?")
                            variant_dds = []
                            with gr.Accordion("Elegir la versión de cada cambio", open=False):
                                for a in AXIS_NAMES:
                                    opts = list(AXES[a]["variants"])
                                    variant_dds.append(gr.Dropdown(opts, value=opts[0], label=AXES[a]["label"]))
                        custom = gr.Textbox(label="¿Tienes una idea propia? (opcional)",
                                            placeholder="Por ejemplo: luz de luna llena reflejada en el agua", max_lines=2)
                        start_btn = gr.Button("Crear mis cinco versiones", variant="primary", size="lg")
                status = gr.HTML()

                gr.HTML(_step(3, "Marca las que te gustan", "Puedes contarnos por qué; nos ayuda a acertar en la próxima ronda"))
                imgs, caps, reasons, acc_btns, rej_btns, regen_btns = [], [], [], [], [], []
                with gr.Row(equal_height=False):
                    for i in range(N):
                        with gr.Column(min_width=190, elem_classes="pj-card"):
                            imgs.append(gr.Image(show_label=False, interactive=False, height=210,
                                                 buttons=["download", "fullscreen"]))
                            caps.append(gr.HTML())
                            with gr.Row():
                                acc_btns.append(gr.Button("Me gusta", size="sm", elem_classes="pj-like"))
                                rej_btns.append(gr.Button("No me gusta", size="sm"))
                            reasons.append(gr.Textbox(show_label=False, placeholder="¿Por qué? (opcional)", max_lines=2))
                            regen_btns.append(gr.Button("Probar otra versión", size="sm", elem_classes="pj-quiet"))

                with gr.Accordion("¿Te gustan o no todas las que faltan?", open=False):
                    pkg_comment = gr.Textbox(label="Comentario general (opcional)")
                    with gr.Row():
                        acc_all = gr.Button("Me gustan todas las que faltan")
                        rej_all = gr.Button("No me gusta ninguna de las que faltan")

                gr.HTML(_step(4, "Mira lo que aprendimos y sigue", "Cada ronda nueva parte de lo que elegiste"))
                with gr.Row():
                    sum_btn = gr.Button("Ver resumen de mis elecciones", interactive=False, variant="primary")
                    next_btn = gr.Button("Siguiente ronda", interactive=False)
                summary_md = gr.HTML()
                close_btn = gr.Button("Terminar y borrar mis imágenes", size="sm", elem_classes="pj-quiet")

                with gr.Accordion("Detalles técnicos", open=False):
                    tech_md = gr.Markdown()
                    plan_df = gr.Dataframe(headers=["Pos.", "Anterior", "Decisión", "Acción", "Siguiente", "Motivo", "Thompson θ"],
                                           label="Plan de la siguiente ronda (calculado del registro, no por el LLM)",
                                           interactive=False, wrap=True)
                    debug_json = gr.JSON(label="Salida del agente LLM y errores del verificador")

            with gr.Tab("Tu opinión"):
                gr.Markdown("Cuéntanos cómo te fue. 1 = totalmente en desacuerdo · 5 = totalmente de acuerdo.")
                sus = [gr.Radio([1, 2, 3, 4, 5], label=f"{i + 1}. {q}") for i, q in enumerate(metrics.SUS_ITEMS)]
                clarity = gr.Slider(1, 5, value=3, step=1, label="¿Qué tan claro fue el resumen de tus elecciones?")
                fidelity = gr.Slider(1, 5, value=3, step=1, label="¿El resumen reflejó lo que realmente elegiste?")
                s_comment = gr.Textbox(label="Comentarios")
                s_btn = gr.Button("Enviar", variant="primary")
                s_out = gr.Markdown()

            with gr.Tab("Privacidad"):
                gr.Markdown(open(os.path.join(os.path.dirname(__file__), "docs", "etica.md"), encoding="utf-8").read())

            with gr.Tab("Panel del equipo"):
                gr.HTML('<p class="pj-panel-note">Métricas de evaluación del proyecto (no hace falta para usar el estudio).</p>')
                include_sim = gr.Checkbox(False, label="Incluir sesiones simuladas")
                mon_btn = gr.Button("Actualizar métricas")
                mon_md = gr.Markdown()
                with gr.Row():
                    mon_axes = gr.Dataframe(headers=["Eje", "Decisiones", "Aceptación"], label="Aceptación por eje")
                    mon_guard = gr.Dataframe(headers=["Tipo", "Categoría", "N"], label="Bloqueos de guardrails")
                mon_err = gr.JSON(label="Errores más frecuentes del LLM (detectados por el verificador)")

        mode.change(lambda m: gr.update(visible=m == MODE_PICK), [mode], [pick_box])
        refresh_outs = [state, status, sum_btn, tech_md] + [c for i in range(N) for c in (imgs[i], caps[i], reasons[i])]
        start_btn.click(on_start, [inp, mode, axes, custom] + variant_dds,
                        refresh_outs + [summary_md, debug_json, next_btn])
        for i in range(N):
            acc_btns[i].click(lambda s, r, i=i: on_decide(s, i, "aceptada", r), [state, reasons[i]], refresh_outs)
            rej_btns[i].click(lambda s, r, i=i: on_decide(s, i, "rechazada", r), [state, reasons[i]], refresh_outs)
            regen_btns[i].click(lambda s, i=i: on_regenerate(s, i), [state], refresh_outs)
        acc_all.click(lambda s, c: on_bulk(s, "aceptada", c), [state, pkg_comment], refresh_outs)
        rej_all.click(lambda s, c: on_bulk(s, "rechazada", c), [state, pkg_comment], refresh_outs)
        sum_btn.click(on_summarize, [state], [summary_md, debug_json, plan_df, next_btn]).then(
            lambda s: gr.update(interactive=False), [state], [sum_btn])
        next_btn.click(on_next, [state], refresh_outs + [next_btn])
        close_btn.click(on_close, [state], refresh_outs + [summary_md, debug_json, plan_df, next_btn])
        mon_btn.click(on_monitor, [include_sim], [mon_md, mon_axes, mon_guard, mon_err])
        s_btn.click(on_survey, [state, s_comment, clarity, fidelity] + sus, [s_out])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=["mock", "flux", "sd"], default=None)
    ap.add_argument("--llm", choices=["auto", "claude", "ollama", "simulado"], default=None)
    ap.add_argument("--ollama-model", default=None, help="modelo de Ollama (por defecto qwen2.5:3b)")
    ap.add_argument("--share", action="store_true", help="enlace público de Gradio para la demo")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 para acceder por túnel SSH desde otra máquina (p. ej. un nodo de Khipu)")
    args = ap.parse_args()
    if args.generator:
        SETTINGS.generator = args.generator
    if args.llm:
        SETTINGS.llm = args.llm
    if args.ollama_model:
        SETTINGS.ollama_model = args.ollama_model
    # En un clúster compartido, PAISAJIA_APP_PASSWORD protege la interfaz (usuario: paisajia).
    pw = os.getenv("PAISAJIA_APP_PASSWORD")
    build().queue().launch(share=args.share, server_port=args.port, server_name=args.host,
                           auth=("paisajia", pw) if pw else None,
                           theme=THEME, css=CSS, head=HEAD, js=FORCE_LIGHT_JS)
