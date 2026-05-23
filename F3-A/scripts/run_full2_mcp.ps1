param(
    [int]$NumRound = 50,
    [double]$ClientsSampleRatio = 0.1,
    [int]$NeighborhoodSize = 5,
    [int]$BatchSize = 128,
    [double]$LearningRate = 0.01,
    [double]$Reg = 1.0,
    [string]$PreModel = "HF_LOCAL",
    [string]$HfEmbeddingModel = "D:\models\all-MiniLM-L6-v2",
    [string]$ItemAttributeSet = "A"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = "python"
$VenvPython = "D:\EEE\vscode\UFGraphFR\.venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
}

& $Python train.py `
  --alias UFGraphFR `
  --dataset mcp `
  --mcp_data_dir data/mcp_user_split `
  --mcp_user_json data/mcp_prepare/user_side/raw/filtered_clients.json `
  --use_jointembedding true `
  --use_item_attribute true `
  --item_attribute_set $ItemAttributeSet `
  --pre_model $PreModel `
  --hf_embedding_model $HfEmbeddingModel `
  --embed_dim 384 `
  --use_transformer false `
  --dp 0.0 `
  --num_round $NumRound `
  --clients_sample_ratio $ClientsSampleRatio `
  --local_epoch 1 `
  --neighborhood_size $NeighborhoodSize `
  --neighborhood_threshold 1.0 `
  --mp_layers 1 `
  --similarity_metric cosine `
  --lr $LearningRate `
  --lr_eta 1 `
  --batch_size $BatchSize `
  --reg $Reg `
  --num_negative 4
