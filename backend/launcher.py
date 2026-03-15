import uvicorn
import webview
import threading
import time
import os
import sys

# 取得資源路徑 (PyInstaller 專用邏輯)
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def start_server():
    """啟動 FastAPI 後端服務"""
    import main
    uvicorn.run(main.app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    print("="*50)
    print("  旗語辨識教學系統 - 原生視窗啟動器")
    print("="*50)
    
    # 檢查網頁資源
    build_path = get_resource_path(os.path.join("frontend", "build"))
    if not os.path.exists(build_path):
        build_path = os.path.abspath("../frontend/build")

    if not os.path.exists(build_path):
        print(f"[ERROR] 找不到網頁檔案: {build_path}")
        time.sleep(5)
        sys.exit(1)

    print("[INFO] 正在初始化偵測引擎 (YOLO v11)...")

    # 1. 啟動後端
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # 2. 等待初始化
    time.sleep(6)

    # 3. 啟動使用者介面
    print("[INFO] 正在啟動使用者介面...")
    window = webview.create_window(
        '旗語辨識教學系統', 
        'http://127.0.0.1:8000',
        width=1400,
        height=900,
        min_size=(1024, 768),
        background_color='#282c34'
    )
    webview.start()
    sys.exit(0)
