## Ética, guardrails y privacidad

**Guardrails implementados**

- **Entrada:** se descartan fotos con rostros o marcas de agua y se eliminan los metadatos EXIF, incluida la ubicación GPS.
- **Instrucciones del usuario:** se bloquean pedidos de añadir personas, texto, construcciones, contenido sexual o violento,
  desastres sobre lugares reales o la eliminación de la etiqueta de IA.
- **Salida:** cada imagen se revisa (detector de rostros + CLIP) antes de mostrarla y se descarta si inserta personas, texto o
  contenido no apto; se reintenta con otra semilla y, si persiste, se prueba otra variante del mismo eje.
- **Transparencia:** toda variación lleva la etiqueta visible «Generado con IA» y metadatos de procedencia en el PNG
  (`paisajia:provenance`, tipo de fuente IPTC *trainedAlgorithmicMedia*, modelo, semilla).
- **Privacidad:** el registro guarda el hash SHA-256 de la foto, nunca la foto. Al terminar la sesión se borran las imágenes
  generadas. El agente LLM recibe solo el registro (IDs, ejes, variantes, decisiones y motivos), nunca la imagen.

**Reflexión.** Una escena alterada de un lugar real puede presentarse como auténtica en turismo o en noticias (p. ej. un
mirador «con nieve» que nunca la tiene). Por eso etiquetamos todo y bloqueamos desastres ficticios. Los filtros basados en
CLIP y en palabras clave pueden fallar en ambos sentidos: dejar pasar casos o bloquear casos legítimos. Por eso la decisión
final siempre es del usuario y reportamos los bloqueos como métrica en la pestaña *Monitoreo*.

**Sesgos.** Los modelos preentrenados reproducen la estética dominante de sus datos (paisajes del hemisferio norte,
«atardecer de postal», estilos artísticos europeos). Las variantes predefinidas también reflejan elecciones culturales del
equipo. Medimos la aceptación por eje y por tipo de escena para detectar si alguna escena (p. ej. desierto) se degrada más.

**Marco de referencia:** Reglamento de la Ley N.º 31814 (el sistema es de riesgo aceptable; aplicamos supervisión humana y
transparencia), Ley N.º 29733 de Protección de Datos Personales y artículo 50 del AI Act europeo sobre contenido sintético.
