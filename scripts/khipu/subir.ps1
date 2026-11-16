# Sube PaisajIA a Khipu desde Windows (PowerShell). Pedirá tu contraseña de Khipu dos veces (scp y ssh).
#     .\scripts\khipu\subir.ps1 -Usuario nombre.apellido
#     .\scripts\khipu\subir.ps1 -Usuario nombre.apellido -ConDatos   # incluye data/lhq_subset para la evaluación
# Solo se suben las rutas listadas abajo: credenciales (.env, Kiphu.txt), .venv y resultados locales NUNCA salen de tu PC.
param(
    [Parameter(Mandatory = $true)][string]$Usuario,
    [switch]$ConDatos,
    [string]$Servidor = "khipu.utec.edu.pe"
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$items = @("app.py", "paisajia", "scripts", "tests", "docs", "examples", "README.md", "COLABORADORES.md",
           "requirements.txt", "requirements-gpu.txt", ".env.example", ".gitignore")
if ($ConDatos) {
    if (-not (Test-Path (Join-Path $root "data\lhq_subset"))) { throw "No existe data\lhq_subset (ejecuta scripts/prepare_dataset.py)" }
    $items += "data/lhq_subset"
}
$tgz = Join-Path $env:TEMP "paisajia.tgz"
Push-Location $root
try {
    tar -czf $tgz --exclude "__pycache__" --exclude "*.pyc" @items
    if ($LASTEXITCODE -ne 0) { throw "tar falló" }
} finally { Pop-Location }
Write-Host ("Paquete: {0:N1} MB" -f ((Get-Item $tgz).Length / 1MB))

scp $tgz "${Usuario}@${Servidor}:paisajia.tgz"
if ($LASTEXITCODE -ne 0) { throw "scp falló" }
# Se extrae encima: actualiza el código sin tocar .venv, modelos ni resultados que ya estén en Khipu.
ssh "${Usuario}@${Servidor}" "mkdir -p ~/paisajia/logs && tar -xzf ~/paisajia.tgz -C ~/paisajia && rm ~/paisajia.tgz && sed -i 's/\r$//' ~/paisajia/scripts/khipu/*.sh ~/paisajia/scripts/khipu/*.sbatch && echo 'Subido a ~/paisajia'"
Remove-Item $tgz
