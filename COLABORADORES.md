# Guía para colaboradores de PaisajIA

Esta guía explica cómo poner en marcha el proyecto desde cero (en tu PC, en Google Colab o en Khipu) y qué reglas seguir para
trabajar en equipo. El detalle técnico (arquitectura, agente, verificador, evaluación) está en el [README](README.md).
Las secciones 1 a 3 bastan para empezar; la 4 y la 5 son para generar imágenes reales.

## 1. Qué es y dónde se ejecuta

PaisajIA recibe una foto de paisaje y genera **cinco variaciones** (hora, estilo, estación, clima, época). El usuario marca
cada una con «Me gusta» o «No me gusta». Luego un **agente LLM** resume las decisiones, un **verificador** comprueba ese
resumen y un **planificador (Thompson)** arma la siguiente ronda. La especificación oficial es `Proyecto2-GrupoMitsuo.pdf`
(v2) y la rúbrica, `EvaluaciónProyecto2 (1).pdf`; ambos se comparten aparte, no están en el repositorio.

El proyecto corre en tres sitios:

| Dónde | Para qué | Generador de imágenes | Agente |
|---|---|---|---|
| **Tu PC** (sin GPU grande) | Desarrollar la interfaz, el registro, el agente, el planificador y las pruebas | `mock` (filtros simples, sin GPU) | `simulado` o Qwen local |
| **Google Colab** (GPU T4 gratis) | Demo real y pruebas con los modelos, sin cuenta de Khipu | FLUX.2 [klein] (principal) o SD + ControlNet (línea base) | Claude (con clave), `simulado` o Qwen |
| **Khipu** (clúster de UTEC) | Evaluación completa del informe (sesiones largas, A6000/A100) | FLUX.2 [klein] o SD + ControlNet | Qwen local |

FLUX necesita una GPU de 16 GB, así que **las imágenes reales se generan en Colab o en Khipu**, no en una laptop.

## 2. Obtener el proyecto

El código está en GitHub: https://github.com/MitsuoMurakami/Proyecto2_Planificacion

```powershell
git clone https://github.com/MitsuoMurakami/Proyecto2_Planificacion.git
cd Proyecto2_Planificacion
```

Si no usas git, descarga el zip desde GitHub (*Code → Download ZIP*) y descomprímelo.

## 3. Instalación en tu PC

Requisitos: Python 3.10 o superior. En Windows usa PowerShell; en Linux/macOS, la terminal.

```powershell
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # Linux/macOS: cp .env.example .env
pytest -q                              # deben pasar todas las pruebas
```

Para abrir la app:

```powershell
python app.py                          # modo mock; abre http://localhost:7860
python app.py --llm simulado           # agente sin conexión
```

Hay una foto de prueba en `examples/paisaje_demo.jpg`. La app tiene cuatro pestañas: **Estudio** (el flujo de 4 pasos),
**Tu opinión** (encuesta SUS), **Privacidad** y **Panel del equipo** (métricas). Los datos técnicos de cada ronda
(similitud, LPIPS, plan de Thompson, JSON del agente) están en el desplegable **Detalles técnicos** del Estudio.

Para que el agente use Qwen en tu PC (opcional, ~2 GB):

1. Instala Ollama desde https://ollama.com/download (Windows: `winget install Ollama.Ollama`).
2. Ejecuta `ollama pull qwen2.5:3b`.
3. En tu `.env`, pon `PAISAJIA_LLM=ollama`.

## 4. Uso en Google Colab (GPU T4)

Es la forma más rápida de ver FLUX funcionando. No necesitas instalar nada en tu PC.

1. Abre [notebooks/PaisajIA_Colab_T4.ipynb](notebooks/PaisajIA_Colab_T4.ipynb) en Colab: en
   https://colab.research.google.com ve a *Archivo → Abrir cuaderno → GitHub* y pega la URL del repositorio.
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → **GPU T4***.
3. En **Secretos** (icono de llave 🔑, a la izquierda) agrega las claves que vayas a usar y activa el acceso del notebook:
   - `PAISAJIA_APP_PASSWORD`: **recomendada**. El enlace público de Gradio queda protegido (usuario `paisajia`);
   - `HF_TOKEN`: si Hugging Face pide aceptar la licencia del modelo;
   - `ANTHROPIC_API_KEY`: opcional, para el agente con Claude (sin ella se usa el agente `simulado`).
4. Ejecuta las celdas **en orden**:

| Paso | Qué hace | Tiempo aprox. |
|---|---|---|
| 1 · GPU | Comprueba que tienes una T4 | segundos |
| 2 · Código | Clona el repositorio (`FUENTE = "git"`) | segundos |
| 3 · Instalación | Instala las dependencias sin tocar el `torch` de Colab (`USAR_OLLAMA = True` para Qwen) | 2–4 min |
| 4 · Configuración | Lee los Secretos y aplica el perfil T4 | segundos |
| 5 · Modelos | Descarga FLUX.2 [klein] 4B, DINOv2, CLIP y LPIPS | ~10 min |
| 6 · Prueba rápida | Pruebas + una ronda con FLUX; muestra las imágenes, el tiempo y la VRAM | 2–5 min |
| 7 · App | Lanza la interfaz y da un enlace `https://….gradio.live` | hasta que la detengas |
| 8 · Evaluación | (opcional) FLUX vs SD sobre `data/lhq_subset` | ~1–2 min por imagen |
| 9 · Resultados | Copia `results/` a tu Google Drive | segundos |

**Perfil T4** (lo aplica el notebook): fp16, generación a 768 px y el codificador de texto de FLUX en 4 bits
(`PAISAJIA_TE_QUANT=4bit`). Así FLUX cabe entero en los 15 GB de la GPU. Sin la cuantización habría que descargar FLUX a la
CPU (~16 GB de RAM), y Colab gratis solo da ~12.7 GB.

**A tener en cuenta en Colab:**

- Si el repositorio es privado, Colab no puede clonarlo. En ese caso ejecuta `.\scripts\colab\empaquetar.ps1` en tu PC,
  sube `paisajia_colab.zip` a *Mi unidad* de Drive y usa `FUENTE = "drive"`.
- Todo lo que está en `/content` se borra al cerrar la sesión: los modelos se vuelven a descargar en cada sesión. Para no
  perder el registro de decisiones pon `GUARDAR_EN_DRIVE = True` (paso 4), y guarda los resultados con el paso 9.
- Las sesiones gratis duran unas horas y se cortan si estás inactivo. Para la evaluación usa `--limit`.
- Si alguna imagen sale negra (desborde en fp16), prueba otra semilla con «Probar otra versión» o `PAISAJIA_GEN_SIZE=640`.
- Nunca escribas claves en las celdas del notebook: usa los Secretos.

## 5. Uso en Khipu

Cada integrante usa **su propia cuenta de Khipu** e instala el proyecto en su carpeta personal. Necesitarás unos 40 GB libres;
compruébalo con `df -h ~` o `quota -s`. Los pasos completos están en el README («Ejecución en Khipu»); en resumen:

1. **Subir el código** (desde tu PC, PowerShell):
   `.\scripts\khipu\subir.ps1 -Usuario <tu_usuario>`. Pide tu contraseña dos veces.
   Si Windows bloquea el script, usa `powershell -ExecutionPolicy Bypass -File .\scripts\khipu\subir.ps1 -Usuario <tu_usuario>`.
2. **Instalar una vez** (en Khipu, nodo de acceso, dentro de `screen -S setup`):
   `bash ~/paisajia/scripts/khipu/setup.sh`.
3. **Prueba rápida**: `cd ~/paisajia && sbatch scripts/khipu/smoke.sbatch`, y luego `tail -f logs/smoke-<número>.out`.
   Usa el número que imprime `sbatch`.
4. **App**: `sbatch scripts/khipu/app.sbatch`. El log indica el comando del túnel SSH; con él abierto, ve a http://localhost:7860.
5. **Evaluación**: `sbatch scripts/khipu/eval.sbatch`. Descarga los resultados con `scp -r`, porque Khipu no tiene backups.

Comandos útiles: `squeue -u $USER` (estado: `PD` en cola, `R` ejecutando), `scancel <número>` (detener) y
`myaccount` (tus límites).

**Reglas de Khipu** (su política de uso):

- Nunca ejecutes la app ni los modelos en el nodo de acceso; todo va por `sbatch`/`srun`.
- Con 3 contraseñas incorrectas se bloquea tu IP.
- Uso solo educativo y de investigación.

Si varios integrantes van a usar Khipu, pidan a los administradores (khipu@utec.edu.pe) una carpeta de grupo. Así los ~35 GB
de modelos se descargan una sola vez: basta con apuntar `HF_HOME`, `TORCH_HOME` y `OLLAMA_MODELS` a esa carpeta antes de
`setup.sh` y de cada `sbatch`.

## 6. Qué no se comparte

| No compartir | Motivo |
|---|---|
| Carpetas personales (p. ej. `Mitsuo/`) | Credenciales y datos locales de cada integrante |
| `.env` y claves (Anthropic, Hugging Face) | Son de cada uno; comparte solo `.env.example`. En Colab van en los Secretos |
| Usuarios y contraseñas de Khipu | Son personales; cada uno usa su cuenta |
| `.venv/` | Depende de cada PC; cada uno crea el suyo con `requirements.txt` |
| `data/` (SQLite y sesiones) | Registros de sesiones de prueba |
| Fotos de voluntarios y variaciones de LHQ | El PDF exige autorización escrita y no publicar variaciones |

El `.gitignore` ya excluye todo lo de la tabla, así que `git add .` es seguro. Antes de cada commit revisa `git status`: no
deben aparecer `.venv`, `.env` ni `data/`.

## 7. Organización del código

| Si trabajas en… | Archivos principales | Responsable (plan del PDF) |
|---|---|---|
| Preprocesamiento y FLUX | `paisajia/preprocessing.py`, `paisajia/generation/flux.py`, `paisajia/quality.py` | Oscar Chu |
| Línea base SD + ControlNet | `paisajia/generation/sd_controlnet.py`, `scripts/eval_generation.py` | Mitsuo Murakami |
| Interfaz, registro y guardrails | `app.py`, `paisajia/registry.py`, `paisajia/guardrails.py` | Christian Frisancho |
| Agente, verificador y planificación | `paisajia/agent/`, `paisajia/planner.py` | Luis Jauregui |
| Métricas, usuarios e informe | `paisajia/metrics.py`, `scripts/eval_*.py`, pestaña Tu opinión | Gonzalo Rodriguez |
| Ejecución en Colab / Khipu | `notebooks/`, `scripts/colab/`, `scripts/khipu/`, `requirements-*.txt` | Todo el equipo |

La configuración compartida (ejes, variantes, umbrales y modelos) está en `paisajia/config.py`. Si cambias un umbral o una
variante, avisa al equipo: afecta a las métricas del informe.

## 8. Antes de entregar un cambio

1. Corre `pytest -q`: todas las pruebas deben pasar.
2. Prueba el flujo en la app en modo `mock`: subir foto → marcar las 5 → ver resumen → siguiente ronda.
3. Si tocaste la generación, corre el paso 6 del notebook de Colab (o `smoke.sbatch` en Khipu) y revisa `vram_pico_gb` y
   `segundos_ronda`.
4. Si cambiaste cómo se ejecuta algo, actualiza el README y esta guía.

## 9. Estado actual

**Listo y probado en PC:**

- Los 7 pasos del flujo, la interfaz (rediseñada: fondo blanco, flujo de 4 pasos, lenguaje para el usuario), el registro,
  el verificador, el planificador y los scripts de evaluación.
- La integración con Qwen vía Ollama, probada contra un servidor de prueba.

**Pendiente de validar con GPU (Colab o Khipu):**

- FLUX con el codificador de texto en 4 bits en la T4: que cargue sin agotar la RAM, la VRAM real y el tiempo por ronda
  (primera prueba: paso 6 del notebook).
- SD + ControlNet con los modelos reales.
- Qwen real en las 100 sesiones de `eval_agent.py`.

**Pendiente del equipo:**

- Descargar LHQ1024 de Kaggle y correr `scripts/prepare_dataset.py`.
- Reclutar 6–8 usuarios para la encuesta SUS.
- Escribir el informe (5 páginas) y grabar el video.
