# Raccourcis pour utiliser node/npm depuis le zip extrait
# Usage : . .\scripts\use-node.ps1   (noter le point au début)

$NODE_DIR = "C:\Users\mbarraud_ut\Downloads\node-v24.14.1-win-x64"
$env:PATH  = "$NODE_DIR;$env:PATH"

# Fonction npm qui contourne le blocage PowerShell
function npm {
    & "$NODE_DIR\node.exe" "$NODE_DIR\node_modules\npm\bin\npm-cli.js" @args
}

# Fonction npx qui contourne le blocage PowerShell
function npx {
    & "$NODE_DIR\node.exe" "$NODE_DIR\node_modules\npm\bin\npx-cli.js" @args
}

Write-Host "Node $(& "$NODE_DIR\node.exe" --version) + npm $(& "$NODE_DIR\node.exe" "$NODE_DIR\node_modules\npm\bin\npm-cli.js" --version) charges." -ForegroundColor Green
Write-Host "Commandes disponibles : node, npm, npx" -ForegroundColor Cyan
