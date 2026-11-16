# PaisajIA: estudio de paisaje con feedback humano y explicación mediante agente LLM

UTEC · Planificación y Toma de Decisiones en IA · Proyecto #2
Equipo: Oscar Chu · Mitsuo Murakami · Christian Frisancho · Luis Jauregui · Gonzalo Rodriguez

PaisajIA recibe una foto de paisaje y genera **exactamente cinco variaciones** controladas en cinco ejes (hora del día,
estilo artístico, estación, clima y época) o según una instrucción propia. El usuario acepta o rechaza cada imagen; las
decisiones se registran en SQLite y JSON. Al cerrar cada ronda, un **agente LLM** resume las decisiones y un **verificador
determinista** garantiza que el resumen no contradiga el registro. Con las decisiones, un **bandido multibrazo (muestreo de
Thompson)** planifica la siguiente ronda: repite lo aceptado y explora alternativas a lo rechazado.

```
Foto ─► 1 Preprocesamiento ─► 2 Generación ─► 3 Control de calidad ─► 4 Interacción (Gradio)
        (EXIF, filtros, CLIP)   (FLUX.2 / SD)    (DINOv2, guardrails)        │
                                     ▲                                       ▼
                                     │                                 5 Registro (SQLite + JSON) ─► Monitoreo
                                     │                                       │
                                     └──── 7 Planificación ◄──── 6 Agente LLM + verificador ─► Plantilla (si falla ×2)
                                          (Thompson)              (solo el registro, nunca la foto)
```

## Estructura

| Ruta | Paso del flujo / función |
|---|---|
| [paisajia/config.py](paisajia/config.py) | Ejes, variantes, instrucciones, umbrales y modelos |
| [paisajia/preprocessing.py](paisajia/preprocessing.py) | 1 · EXIF, 1024×1024, nitidez, rostros, marcas de agua, escena (CLIP), hash |
| [paisajia/generation/](paisajia/generation/) | 2 · `flux.py` (FLUX.2 [klein] 4B, principal), `sd_controlnet.py` (SD 1.5 + ControlNet Depth, línea base), `mock.py` (modo sin GPU) |
| [paisajia/quality.py](paisajia/quality.py) | 3 · DINOv2, guardrail de salida, reintento con otra semilla, descarga a CPU si falta memoria |
| [paisajia/vision.py](paisajia/vision.py) | DINOv2, CLIP, LPIPS y detector de rostros (con proxies deterministas si no hay GPU) |
| [app.py](app.py) | 4 · Interfaz Gradio: Estudio, Monitoreo, Encuesta (SUS + Likert), Ética |
| [paisajia/registry.py](paisajia/registry.py) | 5 · Registro SQLite + JSON por ronda (hash de la foto, nunca la foto) |
| [paisajia/agent/](paisajia/agent/) | 6 · `llm.py` (Claude / Qwen local vía Ollama / LLM simulado), `verifier.py`, `template.py`, `explainer.py` |
| [paisajia/planner.py](paisajia/planner.py) | 7 · Bandido multibrazo con muestreo de Thompson |
| [paisajia/guardrails.py](paisajia/guardrails.py) | Guardrail de instrucciones, etiqueta «Generado con IA», metadatos de procedencia |
| [paisajia/metrics.py](paisajia/metrics.py) | Métricas de la pestaña Monitoreo y puntaje SUS |
| [paisajia/session.py](paisajia/session.py) | Orquestador de la sesión (une los pasos 1-7) |
| [scripts/](scripts/) | Dataset (LHQ1024), evaluación de generación, agente y planificador, imagen de demo, descarga de modelos, prueba rápida |
| [scripts/khipu/](scripts/khipu/) | Instalación y trabajos SLURM para el clúster Khipu de UTEC |
| [tests/](tests/) | Pruebas (sin GPU ni red) |
| [docs/etica.md](docs/etica.md) | Ética, guardrails y marco normativo |

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # modo de prueba sin GPU (interfaz, registro, agente, verificador)
pip install -r requirements-gpu.txt      # modelos reales: FLUX.2, SD + ControlNet, DINOv2, CLIP, LPIPS
cp .env.example .env                      # y completa ANTHROPIC_API_KEY para usar Claude
```

> **GPU.** FLUX.2 [klein] 4B necesita ~13 GB de VRAM en bf16 (GPU de 16–24 GB). Con menos memoria se activa
> automáticamente `enable_model_cpu_offload()` (más lento). En una laptop con 4 GB conviene usar el modo `mock`
> para desarrollar y correr los modelos reales en la GPU del laboratorio o en Colab.
>
> FLUX.2 [klein] es reciente: si tu versión de `diffusers` no trae `Flux2KleinPipeline`, instala diffusers desde
> `main` (`pip install git+https://github.com/huggingface/diffusers`). El ID del modelo se cambia con `PAISAJIA_FLUX_ID`.

## Ejecución

```bash
python scripts/make_demo_image.py examples/paisaje_demo.jpg   # foto de prueba sintética (opcional)

python app.py                         # modo sin GPU (mock) y agente según credenciales (auto)
python app.py --generator flux        # método principal
python app.py --generator sd          # línea base
python app.py --llm simulado          # agente sin conexión
python app.py --generator flux --share   # enlace público para la demo
```

Variables de entorno (o `.env`): `PAISAJIA_GENERATOR` (`mock|flux|sd`), `PAISAJIA_LLM` (`auto|claude|ollama|simulado`),
`PAISAJIA_LLM_MODEL` (por defecto `claude-opus-5-5`), `PAISAJIA_OLLAMA_MODEL` (por defecto `qwen2.5:3b`), `OLLAMA_HOST`, `PAISAJIA_VISION=off` (usar proxies, sin descargar modelos),
`PAISAJIA_DATA_DIR`.

### Agente local con Qwen 2.5 3B (Ollama)

Gratis y privado: el registro no sale de la máquina. El modelo cuantizado ocupa ~2 GB (cabe en una GPU de 4 GB o
corre en CPU).

```bash
# 1. Instala Ollama: https://ollama.com/download   (Windows: winget install Ollama.Ollama)
ollama pull qwen2.5:3b
python app.py --llm ollama                       # o PAISAJIA_LLM=ollama en .env
python app.py --llm ollama --ollama-model qwen3:4b   # otro modelo de Ollama
```

Se usa salida estructurada (`format` = JSON schema) y temperatura 0. Si Ollama no está abierto o falta el modelo, el
agente registra el fallo y usa la plantilla determinista.

### Ejecución en Google Colab (GPU T4)

1. Abre [notebooks/PaisajIA_Colab_T4.ipynb](notebooks/PaisajIA_Colab_T4.ipynb) en Colab (*Archivo → Abrir cuaderno → GitHub*) con entorno **GPU T4**.
2. Claves (`HF_TOKEN`, `ANTHROPIC_API_KEY`, `PAISAJIA_APP_PASSWORD`) en los *Secretos* de Colab, nunca en el notebook.
3. Ejecuta las celdas en orden. El código se clona de GitHub (`FUENTE = "git"`). Si el repositorio es privado, ejecuta
   `.\scripts\colab\empaquetar.ps1` en la PC, sube `paisajia_colab.zip` a *Mi unidad* y usa `FUENTE = "drive"`.

La guía paso a paso (tiempos, avisos y problemas comunes) está en [COLABORADORES.md](COLABORADORES.md#4-uso-en-google-colab-gpu-t4).

Perfil T4: fp16, generación a 768 px y el codificador de texto de FLUX en 4 bits (`PAISAJIA_TE_QUANT`, bitsandbytes), así
FLUX cabe entero en los 15 GB de la GPU. Sin cuantizar haría falta descarga a CPU (~16 GB de RAM), y Colab gratis solo da ~12.7 GB.
Las dependencias salen de [requirements-colab.txt](requirements-colab.txt), que conserva el `torch` de Colab.

### Ejecución en Khipu (clúster de UTEC)

Todo corre en Khipu; la laptop solo sube el código y abre la interfaz por un túnel SSH. Los nodos de cómputo no
tienen internet, así que los modelos se descargan una vez en el nodo de acceso. Scripts en [scripts/khipu/](scripts/khipu/).

| Nodo | GPU | Uso recomendado |
|---|---|---|
| g002 | RTX A6000 48 GB | App y evaluación (por defecto en los `.sbatch`) |
| ag001 | 2× A100 40 GB | Alternativa: `sbatch --nodelist=ag001 …` |
| g001 | Tesla T4 16 GB | Prueba rápida (`debug-gpu`); perfil reducido automático: Qwen en CPU, 768 px, fp16 |

```powershell
# 1. Desde tu laptop (PowerShell): sube el código (sin .env, credenciales ni .venv)
.\scripts\khipu\subir.ps1 -Usuario <tu_usuario>              # -ConDatos para incluir data/lhq_subset
```
```bash
# 2. En Khipu, nodo de acceso, UNA vez (~35 GB de modelos; puede tardar):
ssh <tu_usuario>@khipu.utec.edu.pe
bash ~/paisajia/scripts/khipu/setup.sh       # venv + Ollama + qwen2.5:3b + FLUX/SD/DINOv2/CLIP/LPIPS

# 3. Prueba rápida en la T4 (≤ 30 min): tiempo por imagen, pico de VRAM y resumen de Qwen
cd ~/paisajia && sbatch scripts/khipu/smoke.sbatch
cat logs/smoke-<id>.out

# 4. App (hasta 8 h). El log muestra el comando del túnel SSH:
sbatch scripts/khipu/app.sbatch               # GENERATOR=sd: sbatch --export=ALL,GENERATOR=sd …
cat logs/app-<id>.out
```
```powershell
# 5. En tu laptop: túnel (déjalo abierto) y luego http://localhost:7860
ssh -N -L 7860:<nodo>:<puerto> <tu_usuario>@khipu.utec.edu.pe
```
```bash
# 6. Evaluación completa (FLUX, SD, agente Qwen ×100, planificador) y descarga de resultados
sbatch scripts/khipu/eval.sbatch
scp -r <tu_usuario>@khipu.utec.edu.pe:paisajia/results .      # desde la laptop; Khipu no tiene backups
```

Notas: `squeue -u $USER` muestra tus trabajos y `scancel <id>` los detiene. La interfaz escucha en la red interna del
clúster; define `PAISAJIA_APP_PASSWORD` antes del `sbatch` para pedir contraseña (usuario `paisajia`). Nunca ejecutes
la app ni los modelos en el nodo de acceso: la política de Khipu lo prohíbe.

### Uso de la interfaz

1. Sube una foto y elige **paquete completo** (los cinco ejes), **ejes individuales** (con la variante de cada eje) y/o
   una **instrucción propia** (pasa por el guardrail). Siempre se generan cinco imágenes.
2. Acepta o rechaza cada imagen con un motivo opcional, **regenera** una imagen puntual, o acepta/rechaza **todas las
   pendientes** con un comentario para el paquete. Cada imagen muestra *por qué* fue elegida.
3. El botón **Generar resumen** se habilita cuando todas las imágenes tienen decisión. Muestra las listas exactas de
   aceptadas y rechazadas, el resumen del agente, si pasó el verificador (o si se usó la plantilla) y la **tabla de
   planificación**, calculada del registro (no la redacta el LLM).
4. **Siguiente ronda** aplica el plan. **Terminar sesión** borra las imágenes generadas.

## Agente de explicación y verificador

- El agente recibe solo el registro (IDs como `R2-3`, eje, variante, decisión, motivo) y devuelve un JSON
  `{aceptadas, rechazadas, resumen}`. Con Claude se usa salida estructurada (JSON schema), así que el JSON siempre es válido.
- El verificador ([verifier.py](paisajia/agent/verifier.py)) comprueba: (1) JSON válido con claves y tipos exactos;
  (2) listas idénticas al registro; (3) que el resumen no mencione ejes/variantes/IDs inexistentes, no contradiga decisiones
  (análisis de polaridad por oración) y no cite motivos que el usuario no escribió (citas textuales «…» y raíces de motivos).
- Si falla, se reintenta hasta dos veces enviando los errores al LLM; si sigue fallando, se usa la plantilla determinista
  y se registra como fallo del LLM.
- Con Claude se activa el *fallback* del lado del servidor (`fallbacks: "default"`): si el clasificador de seguridad
  declinara la petición, la API la reintenta con otro modelo. Si la cuenta no lo admite, se repite la llamada sin él.

## Evaluación

| Dimensión | Script | Protocolo |
|---|---|---|
| Composición, diversidad, adherencia, tiempo | `python scripts/eval_generation.py --methods flux sd` | 50 imágenes LHQ × 5 ejes, FLUX vs SD |
| Iteración | `python scripts/eval_planner.py` | Usuarios simulados con preferencias ocultas: Thompson vs aleatorio |
| Agente | `python scripts/eval_agent.py [--llm claude\|ollama]` | 100 sesiones simuladas, con y sin verificador |
| Usuarios | pestaña **Encuesta** + **Monitoreo** | SUS, claridad y fidelidad (6–8 usuarios) |

Preparar el dataset: descarga LHQ1024 de Kaggle y ejecuta
`python scripts/prepare_dataset.py --src <carpeta_lhq> --per-scene 10` (filtra licencias que permiten derivados según
`metadata.json`, preprocesa, clasifica con CLIP, elige 10 por escena y guarda el análisis exploratorio en `results/eda.json`).

### Resultados preliminares (sin GPU, reproducibles)

Medidos en este repositorio con los componentes simulados; **no** sustituyen a la evaluación con FLUX/SD y Claude.

- **Agente** (`eval_agent.py`, LLM simulado con 30 % de error inyectado, 100 sesiones): sin verificador, 85 % de listas
  exactas y 29 % de salidas con error o alucinación; con verificador, 100 % de listas exactas (71 % válidas al primer
  intento, 1.31 intentos de media).
- **Planificador** (`eval_planner.py`, 200 usuarios simulados, 4 rondas): aceptación 39 % en la ronda 1 → 55 % en las
  rondas ≥ 2 con Thompson, frente a 38 % → 40 % con planificación aleatoria.

## Pruebas

```bash
pytest -q
```

Las pruebas cubren guardrails, planificador, verificador (errores inyectados y plantilla en 200 registros aleatorios),
preprocesamiento (EXIF, nitidez) y una sesión completa de dos rondas con el registro.

## Ética y privacidad

Ver [docs/etica.md](docs/etica.md): guardrails de entrada, instrucción y salida; etiqueta visible «Generado con IA» y
metadatos de procedencia; hash en lugar de la foto; borrado al terminar la sesión; reflexión sobre lugares reales,
sesgos y fallos de los filtros; Ley N.º 31814 y su reglamento, Ley N.º 29733 y artículo 50 del AI Act.

## Plan de trabajo

| Semana | Entregable | Responsable |
|---|---|---|
| 12 | Problema, diseño, validación de FLUX.2 [klein] en la GPU | Todo el equipo |
| 13 | Preprocesamiento y generación con FLUX | Oscar Chu |
| 13 | Línea base SD + ControlNet y comparación | Mitsuo Murakami |
| 14 | Interfaz Gradio, registro y guardrails | Christian Frisancho |
| 14–15 | Agente, verificador y planificación de rondas | Luis Jauregui |
| 15–16 | Métricas, prueba de usuarios, informe y video | Gonzalo Rodriguez |
