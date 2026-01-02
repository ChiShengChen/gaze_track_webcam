#!/bin/bash
# ETH-XGaze 模型啟動腳本
# 使用 gaze_eth conda 環境

# 設置環境變數
export KMP_DUPLICATE_LIB_OK=TRUE

# 獲取腳本目錄
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 初始化 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gaze_eth

echo "======================================"
echo "  Eye Tracking - ETH-XGaze Model"
echo "======================================"
echo ""

# 檢查命令
case "$1" in
    calibrate)
        echo "開始校準..."
        python main.py calibrate --output "${2:-eth_calibration.yaml}" --validate
        ;;
    demo)
        echo "啟動即時展示..."
        python main.py demo --calibration "${2:-data/calibrations/eth_calibration.yaml}"
        ;;
    record)
        echo "開始錄製..."
        python main.py record --calibration "${2:-data/calibrations/eth_calibration.yaml}" --duration "${3:-60}"
        ;;
    evaluate)
        echo "開始評估..."
        python main.py evaluate --calibration "${2:-data/calibrations/eth_calibration.yaml}" --num-points "${3:-16}"
        ;;
    *)
        echo "用法: $0 {calibrate|demo|record|evaluate} [選項]"
        echo ""
        echo "命令:"
        echo "  calibrate [輸出文件]  - 執行校準 (默認: eth_calibration.yaml)"
        echo "  demo [校準文件]       - 即時展示 (默認: data/calibrations/eth_calibration.yaml)"
        echo "  record [校準文件] [時長] - 錄製數據"
        echo "  evaluate [校準文件] [點數] - 評估精度"
        echo ""
        echo "範例:"
        echo "  $0 calibrate my_calibration.yaml"
        echo "  $0 demo my_calibration.yaml"
        echo "  $0 record my_calibration.yaml 120"
        exit 1
        ;;
esac

