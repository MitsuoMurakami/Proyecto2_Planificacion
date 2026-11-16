# Empaqueta PaisajIA para Google Colab (desde Windows, PowerShell):
#     .\scripts\colab\empaquetar.ps1               # crea paisajia_colab.zip en la carpeta del proyecto
#     .\scripts\colab\empaquetar.ps1 -ConDatos     # incluye data/lhq_subset para la evaluación
# Sube el .zip a tu Google Drive (Mi unidad) o directamente desde el notebook.
# Solo se empaquetan las rutas listadas: credenciales (.env, Mitsuo/, Kiphu.txt), .venv y resultados NUNCA salen de tu PC.
param([switch]$ConDatos)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$items = @("app.py", "paisajia", "scripts", "tests", "docs", "examples", "notebooks", "README.md", "COLABORADORES.md",
           "requirements.txt", "requirements-gpu.txt", "requirements-colab.txt", ".env.example", ".gitignore")
if ($ConDatos) {
    if (-not (Test-Path (Join-Path $root "data\lhq_subset"))) { throw "No existe data\lhq_subset (ejecuta scripts/prepare_dataset.py)" }
    $items += "data/lhq_subset"
}
$zip = Join-Path $root "paisajia_colab.zip"
if (Test-Path $zip) { Remove-Item $zip }
Push-Location $root
try {
    tar -a -cf $zip --exclude "__pycache__" --exclude "*.pyc" @items   # tar de Windows crea .zip según la extensión
    if ($LASTEXITCODE -ne 0) { throw "tar falló" }
} finally { Pop-Location }
Write-Host ("Paquete: {0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
Write-Host "Subelo a Google Drive (Mi unidad) y abre notebooks/PaisajIA_Colab_T4.ipynb en Colab."
