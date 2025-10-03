#!/bin/bash

# 眼球追蹤系統安裝腳本
# 適用於 macOS

echo "=== 眼球追蹤系統安裝腳本 ==="
echo ""

# 檢查Python版本
echo "檢查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "錯誤: 未找到Python3，請先安裝Python 3.8+"
    exit 1
fi

# 建立虛擬環境
echo "建立虛擬環境..."
python3 -m venv venv
source venv/bin/activate

# 升級pip
echo "升級pip..."
pip install --upgrade pip

# 安裝依賴
echo "安裝Python依賴..."
pip install -r requirements.txt

# 檢查並安裝portaudio（如果需要）
echo "檢查音訊支援..."
if ! python3 -c "import sounddevice" 2>/dev/null; then
    echo "安裝portaudio..."
    if command -v brew &> /dev/null; then
        brew install portaudio
        pip install --force-reinstall sounddevice
    else
        echo "警告: 未找到Homebrew，請手動安裝portaudio"
        echo "執行: brew install portaudio"
    fi
fi

# 下載Vosk模型
echo ""
echo "下載Vosk語音模型..."

# 檢查是否有curl或wget
if command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
elif command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
else
    echo "錯誤: 未找到curl或wget，請先安裝其中一個"
    echo "執行: brew install curl"
    exit 1
fi

# 自動下載英文模型（預設）
echo "下載英文模型 (vosk-model-small-en-us-0.15)..."
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    $DOWNLOAD_CMD vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    if [ $? -eq 0 ]; then
        unzip vosk-model-small-en-us-0.15.zip
        rm vosk-model-small-en-us-0.15.zip
        echo "英文模型下載完成"
    else
        echo "警告: 英文模型下載失敗，請手動下載"
    fi
else
    echo "英文模型已存在"
fi

# 可選下載中文模型
echo ""
echo "是否也下載中文模型? (y/n)"
read -p "請輸入選擇: " download_cn

if [ "$download_cn" = "y" ] || [ "$download_cn" = "Y" ]; then
    echo "下載中文模型 (vosk-model-small-cn-0.22)..."
    if [ ! -d "vosk-model-small-cn-0.22" ]; then
        $DOWNLOAD_CMD vosk-model-small-cn-0.22.zip https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
        if [ $? -eq 0 ]; then
            unzip vosk-model-small-cn-0.22.zip
            rm vosk-model-small-cn-0.22.zip
            echo "中文模型下載完成"
        else
            echo "警告: 中文模型下載失敗，請手動下載"
        fi
    else
        echo "中文模型已存在"
    fi
fi

# 設定執行權限
echo "設定執行權限..."
chmod +x gaze_tracker.py
chmod +x gaze_overlay.py

echo ""
echo "=== 安裝完成 ==="
echo ""
echo "使用方法:"
echo "1. 啟動虛擬環境: source venv/bin/activate"
echo "2. 基本版本: python gaze_tracker.py --vosk-model ./vosk-model-small-en-us-0.15"
echo "3. 進階版本: python gaze_overlay.py --vosk-model ./vosk-model-small-en-us-0.15"
echo ""
echo "注意事項:"
echo "- 首次執行時需要授權相機和麥克風權限"
echo "- 確保光線充足且均勻"
echo "- 保持距離螢幕50-70cm"
echo ""
echo "如需幫助，請查看README.md"
