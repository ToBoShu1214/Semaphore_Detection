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
    print(f"[DEBUG] Fetching questions from: {path}")
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
    else:
        print(f"[ERROR] questions.json NOT FOUND at {path}")
    return {"chinese": [], "navy": []}

@app.get("/api/mapping")
async def get_mapping():
    path = get_resource_path("mapping.csv")
    print(f"[DEBUG] Fetching mapping from: {path}")
    full_map = {}
    # 1. 載入中文映射
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2: full_map[row[1].strip()] = row[0].strip()
    
    # 2. 注入英數映射 (對應 yolo_logic.py 中的定義)
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
    
    state = {
        "mode": "practice", "system": "chinese", "video_source": "0", "is_flag": True,
        "session": {
            "new_challenge_string": None, "stop_challenge_mode": False, 
            "correction_target": None, "navy_sub_mode": "ALPHA",
            "exam_stats": {"total_signals": 0, "correct_signals": 0}
        }
    }

    mapping_csv = get_resource_path("mapping.csv")
    flag_model = get_resource_path("flag.pt")
    
    detection_task = None
    
    async def start_stream():
        nonlocal detection_task
        if detection_task: detection_task.cancel()
        
        gen = run_detection(
            video_source_str=state["video_source"],
            current_mode=state["mode"],
            current_system=state["system"],
            is_flag_required=state["is_flag"],
            flag_model_path=flag_model,
            mapping_csv_path=mapping_csv,
            session_state=state["session"]
        )
        
        async def run():
            try:
                for img_bytes, data in gen:
                    await websocket.send_text(json.dumps({
                        "image": base64.b64encode(img_bytes).decode('utf-8'),
                        "data": data
                    }))
                    await asyncio.sleep(0.01)
            except Exception as e: print(f"Stream Error: {e}")
        
        detection_task = asyncio.create_task(run())

    await start_stream()

    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)
            p = cmd.get("payload", {})
            
            if cmd["command"] == "set_mode":
                state["mode"] = p.get("mode", state["mode"])
                state["system"] = p.get("system", state["system"])
                await start_stream()
            elif cmd["command"] == "set_camera":
                state["video_source"] = str(p.get("device_id", "0"))
                await start_stream()
            elif cmd["command"] == "set_challenge_mode":
                if p.get("enabled"):
                    state["session"]["new_challenge_string"] = p.get("chars")
                    state["session"]["challenge_payload"] = {"type": p.get("type", "standard")}
                else:
                    state["session"]["stop_challenge_mode"] = True
            elif cmd["command"] == "set_flag_requirement":
                state["is_flag"] = p.get("required", True)
                await start_stream()

    except WebSocketDisconnect: pass
    finally:
        if detection_task: detection_task.cancel()

app.mount("/", StaticFiles(directory=get_resource_path("frontend/build"), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
