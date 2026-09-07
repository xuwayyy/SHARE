#!/usr/bin/env bash

set -euo pipefail
PYTHON="${PYTHON:-python}"


# ──── SR defaults ────────────────────────────
SR_DATASET="${DATASET:-AgriFood2}" # HeiPor, PaviaUni, AgriFood, Brain1, Brain2, AgriFood2
SR_MODEL="${MODEL:-SHARE}"
SR_LOSS="${LOSS:-surerec}"
SR_FACTOR="${FACTOR:-3}"
SR_SIGMA="${SIGMA:-$(python -c 'print(25/255)')}"
SR_SIGMA_REAL="${SIGMA_REAL:-$(python -c 'print(20/255)')}"
SR_GAIN="${GAIN:-$(python -c 'print(1/25)')}"
SR_NOISE_TYPE="${NOISE_TYPE:-gaussian}" # gaussian poisson gaussian_poisson
SR_LR="${LR:-1e-3}"
SR_EPOCHS="${EPOCHS:-10000}"
SR_ALPHA="${ALPHA:-1}"
SR_BS="${BS:-1}"
SR_MODE="${SR_MODE:-single}"
SR_DATA="${SR_DATA:-glass_tiles_ms.mat}" # glass_tiles_ms.mat fake_and_real_beers_ms.mat
SR_TRANSFORM="${TRANSFORM:-ScaleScale}"
SR_N_TRANS="${N_TRANS:-3}"
SR_PATCH_SIZE="${PATCH_SIZE:-0}"
SR_OFFSET="${OFFSET:-0 0}"
SR_SEED="${SEED:-42}"
SR_RANK="${RANK:-4}"
SR_MEMORY_BLOCKS="${MEMORY_BLOCKS:-256}"
SR_BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
SR_BENCHMARK_RUN_TIME="${BENCHMARK_RUN_TIMES:-10}"
SR_RETAIN_RATIO="${RETAIN_RATIO:-1}"

SR_ARGS="--dataset $SR_DATASET --model $SR_MODEL --loss $SR_LOSS \
  --factor $SR_FACTOR \
  --sigma $SR_SIGMA --sigma_real $SR_SIGMA_REAL \
  --gain $SR_GAIN --noise_type $SR_NOISE_TYPE \
  --lr $SR_LR --alpha $SR_ALPHA --bs $SR_BS \
  --sr_mode $SR_MODE --sr_data_name $SR_DATA \
  --transform $SR_TRANSFORM --n_trans $SR_N_TRANS \
  --patch_size $SR_PATCH_SIZE --offset $SR_OFFSET \
  --seed $SR_SEED \
  --rank $SR_RANK \
  --memory_blocks $SR_MEMORY_BLOCKS \
  --max_steps $SR_BENCHMARK_STEPS \
  --run_time $SR_BENCHMARK_RUN_TIME \
  --retain_ratio $SR_RETAIN_RATIO"

# ──── Inpainting defaults ────────────────────
INP_DATASET="${DATASET:-Chikusei}"
INP_MODEL="${MODEL:-SHARE}" # hyperei SHARE
INP_LOSS="${LOSS:-r2rrec}"
INP_SIGMA="${SIGMA:-$(python -c 'print(25/255)')}"
INP_GAIN="${GAIN:-$(python -c 'print(1/25)')}"
INP_NOISE_TYPE="${NOISE_TYPE:-gaussian}" # gaussian poisson gaussian_poisson
INP_LR="${LR:-1e-3}" # 1e-3 for SHARE, 1e-2 for hyperei
INP_EPOCHS="${EPOCHS:-10000}"
INP_ALPHA="${ALPHA:-1}"
INP_BS="${BS:-1}"
INP_MAT_INDEX="${MAT_INDEX:-4}" # inp mask shape index 1-4
INP_INDEX="${INDEX:-4}"  # Chikusei patches 0-4
INP_TRANSFORM="${TRANSFORM:-InpaintingShiftScale}" # InpaintingShiftScale
INP_N_TRANS="${N_TRANS:-4}"
INP_SEED="${SEED:-3407}" # 42
INP_RANK="${RANK:-8}"
INP_MEMORY_BLOCKS="${MEMORY_BLOCKS:-512}"
INP_BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
INP_BENCHMARK_RUN_TIME="${BENCHMARK_RUN_TIMES:-10}"
INP_RETAIN_RATIO="${RETAIN_RATIO:-1}"

INP_ARGS="--dataset $INP_DATASET --model $INP_MODEL --loss $INP_LOSS \
  --sigma $INP_SIGMA --gain $INP_GAIN --noise_type $INP_NOISE_TYPE \
  --lr $INP_LR --alpha $INP_ALPHA --bs $INP_BS \
  --mat_index $INP_MAT_INDEX --index $INP_INDEX \
  --transform $INP_TRANSFORM --n_trans $INP_N_TRANS \
  --seed $INP_SEED \
  --rank $INP_RANK \
  --memory_blocks $INP_MEMORY_BLOCKS \
  --max_steps $INP_BENCHMARK_STEPS \
  --run_time $INP_BENCHMARK_RUN_TIME \
  --retain_ratio $INP_RETAIN_RATIO"  # ← 新增这一行

# ──── Dispatch ───────────────────────────────
MODE="${1:-help}"
CMD="${2:-help}"

case "$MODE" in

  sr)
    case "$CMD" in
      train)
        echo "▶ SR Train  dataset=$SR_DATASET  x${SR_FACTOR}  loss=$SR_LOSS"
        $PYTHON main_sr.py $SR_ARGS --task sr --epochs $SR_EPOCHS \
          ${CKPT:+--ckpt "$CKPT"}
        ;;
      test)
        echo "▶ SR Test   dataset=$SR_DATASET  x${SR_FACTOR}  loss=$SR_LOSS"
        $PYTHON main_sr.py $SR_ARGS --task test_sr \
          ${CKPT:+--ckpt "$CKPT"}
        ;;
      train_real)
        echo "▶ SR Real Train  dataset=$SR_DATASET  x${SR_FACTOR}"
        $PYTHON main_sr.py $SR_ARGS --task sr_real --epochs $SR_EPOCHS \
          ${CKPT:+--ckpt "$CKPT"}
        ;;
      test_real)
        echo "▶ SR Real Test   dataset=$SR_DATASET  x${SR_FACTOR}"
        $PYTHON main_sr.py $SR_ARGS --task test_sr_real \
          ${CKPT:+--ckpt "$CKPT"}
        ;;
      benchmark)
        echo "▶ SR Benchmark  dataset=$SR_DATASET  x${SR_FACTOR}  steps=$SR_BENCHMARK_STEPS run_times=$SR_BENCHMARK_RUN_TIME"
        $PYTHON main_sr.py $SR_ARGS --task benchmark_sr
        ;;
      *)
        echo "SR commands: train | test | train_real | test_real | benchmark"
        ;;
    esac
    ;;

  inp)
    case "$CMD" in
      train)
        echo "▶ Inpainting Train  dataset=$INP_DATASET  loss=$INP_LOSS  noise=$INP_NOISE_TYPE"
        $PYTHON main_inpainting.py $INP_ARGS --task inpainting --epochs $INP_EPOCHS \
          ${CKPT:+--ckpt "$CKPT"}
        ;;
      test)
        echo "▶ Inpainting Test   dataset=$INP_DATASET  loss=$INP_LOSS  noise=$INP_NOISE_TYPE"
        $PYTHON main_inpainting.py $INP_ARGS --task test_inpainting \
          ${CKPT:+--ckpt "$CKPT"}
        ;;
      benchmark)
        echo "▶ Inpainting Benchmark  dataset=$INP_DATASET  steps=$INP_BENCHMARK_STEPS run times=$INP_BENCHMARK_RUN_TIME"
        $PYTHON main_inpainting.py $INP_ARGS --task benchmark_inpainting
        ;;
      *)
        echo "Inpainting commands: train | test | benchmark"
        ;;
    esac
    ;;

  help|*)
    cat <<HELPEOF
Usage: bash scripts/run.sh <mode> <command>

  sr  train          Train super-resolution
  sr  test           Test super-resolution
  sr  train_real     Train real SR
  sr  test_real      Test real SR
  sr  benchmark      Benchmark SR training speed
  inp train          Train inpainting
  inp test           Test inpainting
  inp benchmark      Benchmark inpainting training speed

Common env overrides:
  DATASET          default: sr=Cave, inp=Chikusei
  MODEL            default: SHARE
  LOSS             default: sr=surerec, inp=surerec
  FACTOR           default: 2   (SR only)
  SIGMA            default: 25/255
  GAIN             default: 1/25
  NOISE_TYPE       default: gaussian  (gaussian | poisson | gaussian_poisson)
  LR               default: 1e-3
  EPOCHS           default: 10000
  SR_MODE          default: single   (SR only)
  SR_DATA          default: fake_and_real_beers_ms.mat  (SR only)
  BENCHMARK_STEPS  default: 100  (benchmark only)
  CKPT             explicit checkpoint path (optional)

Examples:
  DATASET=PaviaUni FACTOR=4 bash scripts/run.sh sr train
  DATASET=Indian MAT_INDEX=1 bash scripts/run.sh inp train
  CKPT=/path/to/best.pth.tar bash scripts/run.sh sr test
  NOISE_TYPE=poisson GAIN=0.04 bash scripts/run.sh sr train
  NOISE_TYPE=poisson GAIN=0.04 bash scripts/run.sh inp train
  NOISE_TYPE=gaussian_poisson bash scripts/run.sh inp train
  BENCHMARK_STEPS=200 bash scripts/run.sh sr benchmark
  BENCHMARK_STEPS=200 bash scripts/run.sh inp benchmark
HELPEOF
    ;;
esac