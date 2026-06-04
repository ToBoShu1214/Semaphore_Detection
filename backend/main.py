from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from yolo_logic import run_detection, load_mapping
import base64
import json
import os
import sys
import csv
import threading
import queue
import time
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Try relative to the script directory (backend)
    backend_path = os.path.join(script_dir, relative_path)
    if os.path.exists(backend_path):
        return backend_path
        
    # 2. Try relative to the current working directory
    base_path = os.path.abspath(".")
    cwd_path = os.path.join(base_path, relative_path)
    if os.path.exists(cwd_path):
        return cwd_path
        
    # 3. Try relative to the project root directory
    root_path = os.path.abspath(os.path.join(script_dir, "..", relative_path))
    if os.path.exists(root_path):
        return root_path
        
    return backend_path

@app.get("/api/questions")
async def get_questions():
    path = get_resource_path("questions.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "chinese": data.get("chinese", []),
                    "navy": data.get("navy", []) 
                }
        except Exception as e:
            print(f"[ERROR] Failed to read questions.json: {e}")
    return {"chinese": [], "navy": []}

@app.get("/api/mapping")
async def get_mapping():
    path = get_resource_path("mapping.csv")
    full_map = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2: full_map[row[1].strip()] = row[0].strip()
    
    navy_defaults = {
        'A':'1','B':'2','C':'3','D':'4','E':'5','F':'6','G':'7','H':'8','I':'9','K':'0',
        'J':'J','L':'L','M':'M','N':'N','O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T',
        'U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z',
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','#':'#'
    }
    for k, v in navy_defaults.items():
        if k not in full_map: full_map[k] = v
    return full_map

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 共用狀態
    state = {
        "mode": "practice", "system": "chinese", "video_source": "0", "is_flag": True, "compute_device": "auto",
        "session": {
            "new_challenge_string": None, "stop_challenge_mode": False, 
            "correction_target": None, "navy_sub_mode": "ALPHA",
            "exam_stats": {"total_signals": 0, "correct_signals": 0}
        }
    }
    
    mapping_csv = get_resource_path("mapping.csv")
    flag_model = get_resource_path("flag.pt")
    
    data_queue = queue.Queue(maxsize=1)
    restart_event = threading.Event()
    stop_event = threading.Event()

    def detection_worker():
        print("[THREAD] Detection worker started")
        while not stop_event.is_set():
            restart_event.clear()
            try:
                # 建立新的偵測產生器
                gen = run_detection(
                    video_source_str=state["video_source"],
                    current_mode=state["mode"],
                    current_system=state["system"],
                    is_flag_required=state["is_flag"],
                    flag_model_path=flag_model,
                    mapping_csv_path=mapping_csv,
                    session_state=state["session"],
                    compute_device_pref=state["compute_device"]
                )
                
                for img_bytes, data in gen:
                    if stop_event.is_set() or restart_event.is_set():
                        break
                    
                    if data_queue.full():
                        try: data_queue.get_nowait()
                        except queue.Empty: pass
                    data_queue.put((img_bytes, data))
            except Exception as e:
                print(f"[THREAD] Detection error: {e}")
                time.sleep(1) # 發生錯誤時稍等再重試
            
            # 如果是因為要重啟而跳出迴圈，稍等一下確保資源釋放
            if restart_event.is_set():
                time.sleep(0.5)
                
        print("[THREAD] Detection worker stopped")

    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    async def sender_task():
        last_send_time = 0
        try:
            while not stop_event.is_set():
                try:
                    # 使用非阻塞方式獲取數據
                    img_bytes, data = data_queue.get(timeout=0.1)
                    
                    now = time.time()
                    if now - last_send_time < 0.03: # 限制約 33 FPS
                        continue
                        
                    await websocket.send_text(json.dumps({
                        "image": base64.b64encode(img_bytes).decode('utf-8'),
                        "data": data
                    }))
                    last_send_time = now
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"[WS] Sender task error: {e}")
                    break
        except asyncio.CancelledError:
            pass

    sender_loop = asyncio.create_task(sender_task())

    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)
            p = cmd.get("payload", {})
            
            needs_restart = False
            
            if cmd["command"] == "set_mode":
                state["mode"] = p.get("mode", state["mode"])
                state["system"] = p.get("system", state["system"])
                needs_restart = True
            elif cmd["command"] == "set_camera":
                new_src = str(p.get("device_id", "0"))
                if state["video_source"] != new_src:
                    state["video_source"] = new_src
                    needs_restart = True
            elif cmd["command"] == "set_compute_device":
                new_dev = p.get("device", "auto")
                if state["compute_device"] != new_dev:
                    state["compute_device"] = new_dev
                    needs_restart = True
            elif cmd["command"] == "set_challenge_mode":
                if p.get("enabled"):
                    state["session"]["new_challenge_string"] = p.get("chars")
                    state["session"]["challenge_payload"] = {"type": p.get("type", "standard")}
                else:
                    state["session"]["stop_challenge_mode"] = True
            elif cmd["command"] == "set_flag_requirement":
                new_req = p.get("required", True)
                if state["is_flag"] != new_req:
                    state["is_flag"] = new_req
                    needs_restart = True
            
            if needs_restart:
                restart_event.set()

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    finally:
        stop_event.set()
        restart_event.set() # 喚醒 worker 跳出 generator
        sender_loop.cancel()
        # 不使用 join 阻塞 async 迴圈，daemon thread 會隨進程結束

app.mount("/", StaticFiles(directory=get_resource_path("frontend/build"), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
