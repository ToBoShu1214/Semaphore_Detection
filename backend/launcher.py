import uvicorn
import webview
import threading
import time
import os
import sys

# 取得資源路徑
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def start_server():
    """啟動 FastAPI 後端服務"""
    try:
        import main
        # 強制綁定 127.0.0.1 避免 IPv6 解析問題
        uvicorn.run(main.app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        print(f"[CRITICAL ERROR] 後端啟動失敗: {e}")

if __name__ == "__main__":
    print("="*50)
    print("  旗語辨識教學系統 - 環境相容性啟動器")
    print("="*50)
    
    build_path = get_resource_path(os.path.join("frontend", "build"))
    if not os.path.exists(build_path):
        build_path = os.path.abspath("../frontend/build")

    if not os.path.exists(build_path):
        print(f"[ERROR] 找不到網頁檔案於: {build_path}")
        time.sleep(5)
        sys.exit(1)

    print("[INFO] 偵測硬體環境中...")
    
    # 啟動後端
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # 在 CPU 環境下，模型載入非常緩慢，建議延長等待
    print("[INFO] 正在載入 AI 模型 (CPU 模式可能需要較長時間)，請稍候...")
    
    # 改為動態檢測，或者是極長的固定等待
    time.sleep(12) 

    print("[INFO] 正在啟動使用者介面...")
    window = webview.create_window(
        '旗語辨識教學系統', 
        'http://127.0.0.1:8000', # 這裡也用 127.0.0.1
        width=1400,
        height=900,
        min_size=(1024, 768),
        background_color='#282c34'
    )
    webview.start()
    sys.exit(0)
