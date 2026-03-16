from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from yolo_logic import run_detection, load_mapping # 載入 load_mapping
import base64
import json
import os
import sys
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 取得資源路徑 (PyInstaller 專用)
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    
    # 開發環境下的路徑處理
    base_path = os.path.abspath(".")
    full_path = os.path.join(base_path, relative_path)
    
    # 如果在 backend 目錄下執行，且找不到路徑，嘗試往上一層找
    if not os.path.exists(full_path):
        parent_path = os.path.abspath(os.path.join(base_path, ".."))
        potential_path = os.path.join(parent_path, relative_path)
        if os.path.exists(potential_path):
            return potential_path
            
    return full_path

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("客戶端連接中...")
    await websocket.accept()
    print("客戶端已連接")

    current_mode = "practice"
    current_system = "chinese"
    target_sequence = None
    start_exam_signal = False
    stop_exam_signal = False
    video_source_str = '0' 
    is_flag_required = True 
    
    # 使用 get_resource_path 取得檔案路徑
    flag_model_path = get_resource_path("flag.pt")
    mapping_csv_path = get_resource_path("mapping.csv")
    
    session_state = {
        "new_challenge_string": None,
        "stop_challenge_mode": False,
        "correction_target": None
    }
    
    # 根據系統初始化對應表
    if current_system == "navy":
        vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        reverse_mapping = {char: list(char) for char in vocab}
    else:
        _, reverse_mapping = load_mapping(mapping_csv_path)

    # 在迴圈外部初始化生成器，但允許其重新初始化
    detection_task = None
    generator_instance = None

    async def start_detection_stream():
        nonlocal generator_instance
        nonlocal detection_task
        nonlocal start_exam_signal
        nonlocal stop_exam_signal
        nonlocal video_source_str
        nonlocal flag_model_path
        nonlocal session_state
        nonlocal current_system
        nonlocal reverse_mapping

        # 根據模式決定要載入哪個對應表
        current_mapping_path = mapping_csv_path
        if current_system == "navy":
            vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            reverse_mapping = {char: list(char) for char in vocab}
        else:
            _, reverse_mapping = load_mapping(mapping_csv_path)

        if detection_task:
            detection_task.cancel()
            await asyncio.sleep(0.1) # 給予一些時間進行清理
        
        try:
            generator_instance = run_detection(
                video_source_str=video_source_str, 
                current_mode=current_mode,
                current_system=current_system,
                target_sequence=target_sequence,
                start_exam_signal=start_exam_signal,
                stop_exam_signal=stop_exam_signal,
                is_flag_required=is_flag_required, # 傳遞旗幟要求
                flag_model_path=flag_model_path, # 傳遞旗幟模型路徑
                mapping_csv_path=current_mapping_path, # 傳遞對應表 CSV 路徑
                session_state=session_state # 傳遞 session_state
            )
        except Exception as e:
            print(f"Error initializing detection generator: {e}")
            await websocket.send_text(json.dumps({"error": f"Error initializing video stream: {e}"}))
            return
        
        async def stream_frames():
            try:
                for frame_bytes, detection_data in generator_instance:
                    base64_image = base64.b64encode(frame_bytes).decode('utf-8')
                    
                    payload = {
                        "image": base64_image,
                        "data": detection_data
                    }
                    
                    await websocket.send_text(json.dumps(payload))
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                print("Detection stream task cancelled.")
            except WebSocketDisconnect:
                print("客戶端已斷開連接")
            except Exception as e:
                print(f"Detection stream error: {e}")
                await websocket.send_text(json.dumps({"error": f"Detection stream error: {e}"}))
            finally:
                print("Detection stream finished.")
                if generator_instance:
                    try:
                        generator_instance.close() 
                    except RuntimeError:
                        pass # 生成器已關閉
        
        detection_task = asyncio.create_task(stream_frames())

    await start_detection_stream() # 啟動初始串流

    try:
        while True:
            message = await websocket.receive_text()
            msg_data = json.loads(message)
            
            if msg_data.get("command") == "set_mode":
                new_mode = msg_data.get("payload", {}).get("mode")
                new_system = msg_data.get("payload", {}).get("system")
                new_target_sequence = msg_data.get("payload", {}).get("target_sequence")
                
                if new_mode and new_mode in ["practice", "exam", "correction"]:
                    current_mode = new_mode
                    if new_system in ["chinese", "navy"]:
                        current_system = new_system
                    target_sequence = new_target_sequence
                    start_exam_signal = False # 模式改變時重置信號
                    stop_exam_signal = False
                    print(f"切換模式至: {current_mode}, 系統: {current_system}, 目標序列: {target_sequence}")
                    await start_detection_stream() # 使用新模式重新啟動串流
            
            elif msg_data.get("command") == "set_challenge_mode":
                payload = msg_data.get("payload", {})
                is_enabled = payload.get("enabled")
                string_to_practice = payload.get("chars")
                c_type = payload.get("type", "standard") # Get challenge type

                if is_enabled:
                    session_state["new_challenge_string"] = string_to_practice
                    session_state["stop_challenge_mode"] = False
                    session_state["challenge_payload"] = {"chars": string_to_practice, "type": c_type}
                    print(f"Challenge mode ({c_type}) enabled for string: {string_to_practice}")
                else:
                    session_state["stop_challenge_mode"] = True
                    session_state["new_challenge_string"] = None
                    print("Challenge mode disabled.")
                
                print("Challenge mode updated, restarting detection stream...")
                await start_detection_stream()

            elif msg_data.get("command") == "set_correction_target":
                payload = msg_data.get("payload", {})
                target_signal = payload.get("signal")
                session_state["correction_target"] = target_signal
                print(f"Correction mode target set to: {target_signal}")
                # Restart stream to enter correction mode immediately
                await start_detection_stream()

            elif msg_data.get("command") == "start_exam":
                if current_mode == "exam":
                    start_exam_signal = True
                    stop_exam_signal = False # 確保停止信號關閉
                    print("收到開始考試指令，重新啟動串流以應用設定。")
                    await start_detection_stream()
            elif msg_data.get("command") == "stop_exam":
                if current_mode == "exam":
                    stop_exam_signal = True
                    start_exam_signal = False # 確保開始信號關閉
                    print("收到停止考試指令，重新啟動串流以應用設定。")
                    await start_detection_stream()
            elif msg_data.get("command") == "set_flag_requirement":
                is_flag_required = msg_data.get("payload", {}).get("required", True)
                print(f"設定旗幟要求為: {is_flag_required}, 正在重啟偵測串流...")
                await start_detection_stream()
            elif msg_data.get("command") == "set_video_source":
                new_video_source = msg_data.get("payload", {}).get("source")
                if new_video_source is not None:
                    video_source_str = new_video_source
                    print(f"切換影像來源至: {video_source_str}")
                    await websocket.send_text(json.dumps({"status": f"Attempting to switch video source to {video_source_str}..."}))
                    await start_detection_stream() # 使用新的影像來源重新啟動串流

    except WebSocketDisconnect:
        print("客戶端已斷開連接")
    except Exception as e:
        print(f"WebSocket message handling error: {e}")
    finally:
        if detection_task:
            detection_task.cancel()
        print("WebSocket connection closed.")

# 在所有 API 路由之後，掛載靜態檔案服務
# 使用 get_resource_path 確保在打包後能找到前端檔案
app.mount("/", StaticFiles(directory=get_resource_path("frontend/build"), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
