#!/bin/bash
# L2CS-Net 權重下載腳本

echo "L2CS-Net 權重下載腳本"
echo "===================="
echo ""

# 創建目錄
WEIGHTS_DIR="$HOME/.l2cs/models"
mkdir -p "$WEIGHTS_DIR"

echo "權重將下載到: $WEIGHTS_DIR"
echo ""

# 檢查是否已存在
if [ -f "$WEIGHTS_DIR/L2CSNet_gaze360.pkl" ]; then
    echo "✅ 權重文件已存在: $WEIGHTS_DIR/L2CSNet_gaze360.pkl"
    exit 0
fi

echo "請手動下載權重文件："
echo ""
echo "1. 訪問 GitHub Releases:"
echo "   https://github.com/Ahmednull/L2CS-Net/releases"
echo ""
echo "2. 下載以下文件之一："
echo "   - L2CSNet_gaze360.pkl (推薦，用於 Gaze360 數據集)"
echo "   - L2CSNet_MPIIGaze.pkl (用於 MPIIGaze 數據集)"
echo ""
echo "3. 將下載的文件放到:"
echo "   $WEIGHTS_DIR/"
echo ""
echo "或者使用 wget（如果知道直接下載鏈接）："
echo "   wget <download_url> -O $WEIGHTS_DIR/L2CSNet_gaze360.pkl"
echo ""

# 嘗試從常見的 GitHub releases 下載（如果可能）
echo "正在嘗試查找下載鏈接..."
echo "（如果自動下載失敗，請手動下載）"
echo ""

# 注意：GitHub releases 的 URL 可能會變化，這裡提供一個示例
# 實際使用時需要從 GitHub releases 頁面獲取最新的下載鏈接

echo "完成後，運行以下命令測試："
echo "  python main.py calibrate --output l2cs_test.yaml"

