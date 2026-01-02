#!/usr/bin/env python3
"""
L2CS-Net 權重下載腳本
嘗試從 GitHub Releases 下載權重文件
"""

import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

def download_file(url, dest_path):
    """下載文件並顯示進度"""
    try:
        print(f"正在下載: {url}")
        print(f"保存到: {dest_path}")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r進度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, show_progress)
        print("\n✅ 下載完成！")
        return True
    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP 錯誤: {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"\n❌ 下載失敗: {e}")
        return False

def main():
    print("L2CS-Net 權重下載工具")
    print("=" * 50)
    print()
    
    # 創建目錄
    weights_dir = Path.home() / '.l2cs' / 'models'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    weights_file = weights_dir / 'L2CSNet_gaze360.pkl'
    
    # 檢查是否已存在
    if weights_file.exists():
        size_mb = weights_file.stat().st_size / 1024 / 1024
        print(f"✅ 權重文件已存在: {weights_file}")
        print(f"   大小: {size_mb:.2f} MB")
        response = input("\n是否重新下載？(y/N): ")
        if response.lower() != 'y':
            print("跳過下載")
            return
        weights_file.unlink()
    
    print("\n⚠️  注意：GitHub Releases 的 URL 可能會變化")
    print("請訪問以下鏈接獲取最新的下載地址：")
    print("  https://github.com/Ahmednull/L2CS-Net/releases")
    print()
    
    # 常見的下載 URL 模式（可能已過時）
    possible_urls = [
        "https://github.com/Ahmednull/L2CS-Net/releases/download/v1.0/L2CSNet_gaze360.pkl",
        "https://github.com/Ahmednull/L2CS-Net/releases/latest/download/L2CSNet_gaze360.pkl",
    ]
    
    print("嘗試從常見 URL 下載...")
    success = False
    for url in possible_urls:
        print(f"\n嘗試: {url}")
        if download_file(url, weights_file):
            success = True
            break
        print("失敗，嘗試下一個...")
    
    if not success:
        print("\n" + "=" * 50)
        print("自動下載失敗")
        print("=" * 50)
        print("\n請手動下載：")
        print("1. 訪問: https://github.com/Ahmednull/L2CS-Net/releases")
        print("2. 找到最新的 release")
        print("3. 下載 L2CSNet_gaze360.pkl")
        print(f"4. 放到: {weights_file}")
        print()
        print("或者使用 wget/curl（如果知道直接鏈接）：")
        print(f"  wget <url> -O {weights_file}")
        print(f"  curl -L <url> -o {weights_file}")
    
    # 驗證文件
    if weights_file.exists():
        size_mb = weights_file.stat().st_size / 1024 / 1024
        print(f"\n✅ 權重文件已就緒: {weights_file}")
        print(f"   大小: {size_mb:.2f} MB")
        if size_mb < 1:
            print("   ⚠️  文件大小異常小，可能下載不完整")
        else:
            print("   可以開始使用 L2CS 了！")

if __name__ == '__main__':
    main()
