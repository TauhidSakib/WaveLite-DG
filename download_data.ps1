$ErrorActionPreference = "Continue"
$root = "D:\AI-Projects\WaveMamba"
$data = "$root\data"
$kag  = "$data\_kaggle"
New-Item -ItemType Directory -Force -Path $data,$kag | Out-Null

# Kaggle basic-auth creds
$kj = Get-Content "C:\Users\Asus\.kaggle\kaggle.json" -Raw | ConvertFrom-Json
$cred = "$($kj.username):$($kj.key)"

function Log($m) { $ts = Get-Date -Format "HH:mm:ss"; Add-Content "$root\download.log" "[$ts] $m"; Write-Output "[$ts] $m" }

Set-Content "$root\download.log" "download started"

# ---------- 1. Kvasir-SEG (direct, no auth) ----------
$kvZip = "$data\kvasir-seg.zip"
if (-not (Test-Path "$data\Kvasir-SEG\images")) {
  Log "downloading Kvasir-SEG ..."
  curl.exe -L --fail -o $kvZip "https://datasets.simula.no/downloads/kvasir-seg.zip"
  Log "extracting Kvasir-SEG ..."
  Expand-Archive -Path $kvZip -DestinationPath $data -Force
  Remove-Item $kvZip -Force
  Log "Kvasir-SEG done."
} else { Log "Kvasir-SEG already present." }

# ---------- 2. CVC-ClinicDB (kaggle) ----------
$ccZip = "$kag\cvcclinicdb.zip"; $ccDir = "$kag\cvcclinicdb"
if (-not (Test-Path $ccDir)) {
  Log "downloading CVC-ClinicDB ..."
  curl.exe -L --fail -u $cred -o $ccZip "https://www.kaggle.com/api/v1/datasets/download/balraj98/cvcclinicdb"
  Log "extracting CVC-ClinicDB ..."
  New-Item -ItemType Directory -Force -Path $ccDir | Out-Null
  Expand-Archive -Path $ccZip -DestinationPath $ccDir -Force
  Remove-Item $ccZip -Force
  Log "CVC-ClinicDB done."
} else { Log "CVC-ClinicDB already present." }

# ---------- 3. PraNet test bundle (kaggle) — ColonDB/ETIS/CVC-300 ----------
$prZip = "$kag\pranet.zip"; $prDir = "$kag\pranet"
if (-not (Test-Path $prDir)) {
  Log "downloading PraNet test bundle ..."
  curl.exe -L --fail -u $cred -o $prZip "https://www.kaggle.com/api/v1/datasets/download/debeshjha1/pranet-testdataset"
  Log "extracting PraNet test bundle ..."
  New-Item -ItemType Directory -Force -Path $prDir | Out-Null
  Expand-Archive -Path $prZip -DestinationPath $prDir -Force
  Remove-Item $prZip -Force
  Log "PraNet bundle done."
} else { Log "PraNet bundle already present." }

Log "ALL_DATASETS_DONE"