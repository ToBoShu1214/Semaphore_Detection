import uvicorn
import webview
import threading
import time
import os
import sys
import requests
import json
import ctypes

# 取得資源路徑
def get_resource_path(relative_path):
    # 1. 優先檢查執行檔同級目錄 (方便打包後修改設定)
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        local_path = os.path.join(exe_dir, relative_path)
        if os.path.exists(local_path):
            return local_path

    # 2. 檢查 PyInstaller 內部暫存目錄
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    
    return os.path.join(os.path.abspath("."), relative_path)

def handle_console_visibility():
    """根據 config.json 動態控制控制台視窗的顯示/隱藏"""
    try:
        config_path = get_resource_path('config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            show_console = config.get('SHOW_CONSOLE', True)
            
            if os.name == 'nt':
                whnd = ctypes.windll.kernel32.GetConsoleWindow()
                if whnd != 0:
                    if show_console:
                        ctypes.windll.user32.ShowWindow(whnd, 5) # 5 = SW_SHOW
                    else:
                        ctypes.windll.user32.ShowWindow(whnd, 0) # 0 = SW_HIDE
    except Exception as e:
        print(f"[DEBUG] Console visibility control failed: {e}")

def wait_for_server():
    """不斷檢查後端是否啟動成功"""
    url = "http://127.0.0.1:8000/"
    max_tries = 30 # 最多等 30 秒
    tries = 0
    while tries < max_tries:
        try:
            # 嘗試連接後端
            response = requests.get(url, timeout=1)
            if response.status_code == 200 or response.status_code == 404:
                print(f"[INFO] 後端伺服器已就緒 (耗時 {tries}s)")
                return True
        except:
            pass
        time.sleep(1)
        tries += 1
    return False

def start_server():
    """啟動 FastAPI 後端服務"""
    try:
        import main
        # 使用 0.0.0.0 以增加相容性
        uvicorn.run(main.app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        # 將錯誤寫入日誌檔，方便在打包後偵錯
        with open("backend_error.log", "w") as f:
            f.write(f"Error: {e}")
        print(f"[ERROR] 後端啟動失敗: {e}")

if __name__ == "__main__":
    # 0. 處理控制台顯示邏輯
    handle_console_visibility()

    # 資源檢查
    # 優先嘗試 _MEIPASS (打包後)
    if hasattr(sys, '_MEIPASS'):
        build_path = os.path.join(sys._MEIPASS, "frontend", "build")
    else:
        # 開發環境
        build_path = os.path.abspath("../frontend/build")
        if not os.path.exists(build_path):
            build_path = os.path.join(os.path.abspath("."), "frontend", "build")

    print(f"[DEBUG] 正在尋找前端 Build 路徑: {build_path}")
    if not os.path.exists(build_path):
        print(f"[ERROR] 找不到前端靜態檔案: {build_path}")
        # 在開發環境給予明確提示
        sys.exit(1)

    # 1. 啟動後端執行緒
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # 2. 智慧等待後端連線 (檢查本機地址)
    server_ready = wait_for_server()

    # 3. 啟動使用者介面 (使用 localhost 載入網頁)
    window = webview.create_window(
        '旗語辨識教學系統', 
        'http://127.0.0.1:8000',
        width=1400,
        height=900,
        min_size=(1024, 768),
        background_color='#282c34'
    )
    
    if server_ready:
        webview.start()
    else:
        print("[CRITICAL] 後端啟動逾時，系統無法開啟。")
        sys.exit(1)
