$ErrorActionPreference = "Continue"
$root = "D:\AI-Projects\WaveMamba"
$kag  = "$root\data\_kaggle"
New-Item -ItemType Directory -Force -Path $kag | Out-Null
$kj = Get-Content "C:\Users\Asus\.kaggle\kaggle.json" -Raw | ConvertFrom-Json
$cred = "$($kj.username):$($kj.key)"
function Log($m){ $ts=Get-Date -Format "HH:mm:ss"; Add-Content "$root\download_unseen.log" "[$ts] $m"; Write-Output "[$ts] $m" }
Set-Content "$root\download_unseen.log" "unseen download started"

$sets = @(
  @{ name="cvc-colondb"; slug="longvil/cvc-colondb" },
  @{ name="etis-larib";  slug="nguyenvoquocduong/etis-laribpolypdb" },
  @{ name="cvc-300";     slug="nourabentaher/cvc-300" }
)
foreach ($s in $sets) {
  $dir = "$kag\$($s.name)"; $zip = "$kag\$($s.name).zip"
  if (Test-Path $dir) { Log "$($s.name) already present."; continue }
  Log "downloading $($s.name) <- $($s.slug) ..."
  curl.exe -s -L --fail -u $cred -o $zip "https://www.kaggle.com/api/v1/datasets/download/$($s.slug)"
  if (-not (Test-Path $zip) -or (Get-Item $zip).Length -lt 10000) { Log "!! $($s.name) FAILED (no/small zip)"; continue }
  Log "extracting $($s.name) ..."
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Expand-Archive -Path $zip -DestinationPath $dir -Force
  Remove-Item $zip -Force
  Log "$($s.name) done."
}
Log "UNSEEN_DONE"