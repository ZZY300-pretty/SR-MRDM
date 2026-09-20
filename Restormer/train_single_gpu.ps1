$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$env:CUDA_VISIBLE_DEVICES = "0"
$config = "Deraining/Options/CloudRemoval_Restormer_CloudData_RTX3090.yml"

Write-Host "Working directory: $scriptDir"
Write-Host "Using config: $config"
Write-Host "CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"

python basicsr/train.py -opt $config --launcher none
