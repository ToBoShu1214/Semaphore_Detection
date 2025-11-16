from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from yolo_logic import run_detection
import yolo_logic # 導入 yolo_logic 以設置全域變數
import base64
import json
import os # 導入 os 模組以檢查檔案是否存在
from fastapi.staticfiles import StaticFiles # 導入 StaticFiles

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("客戶端連接中...")
    await websocket.accept()
    print("客戶端已連接")

    current_mode = "practice"
    target_sequence = None
    start_exam_signal = False
    stop_exam_signal = False
    video_source_str = '0' # 預設視訊來源
    flag_model_path = os.path.join(os.path.dirname(__file__), "flag.onnx") # 旗幟模型路徑
    mapping_csv_path = os.path.join(os.path.dirname(__file__), "mapping.csv") # 對應表 CSV 路徑
    
    # 在迴圈外部初始化生成器，但允許其重新初始化
    detection_task = None
    generator_instance = None

    async def start_detection_stream():
        nonlocal generator_instance
        nonlocal detection_task
        nonlocal start_exam_signal
        nonlocal stop_exam_signal
        nonlocal video_source_str
        nonlocal flag_model_path # 宣告 flag_model_path 為 nonlocal

        if detection_task:
            detection_task.cancel()
            await asyncio.sleep(0.1) # 給予一些時間進行清理
        
        try:
            generator_instance = run_detection(
                video_source_str=video_source_str, 
                current_mode=current_mode, 
                target_sequence=target_sequence,
                start_exam_signal=start_exam_signal,
                stop_exam_signal=stop_exam_signal,
                flag_model_path=flag_model_path, # 傳遞旗幟模型路徑
                mapping_csv_path=mapping_csv_path # 傳遞對應表 CSV 路徑
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
                new_target_sequence = msg_data.get("payload", {}).get("target_sequence")
                
                if new_mode and new_mode in ["practice", "exam"]:
                    current_mode = new_mode
                    target_sequence = new_target_sequence
                    start_exam_signal = False # 模式改變時重置信號
                    stop_exam_signal = False
                    print(f"切換模式至: {current_mode}, 目標序列: {target_sequence}")
                    await start_detection_stream() # 使用新模式重新啟動串流
            
            elif msg_data.get("command") == "set_challenge_mode":
                payload = msg_data.get("payload", {})
                is_enabled = payload.get("enabled")
                string_to_practice = payload.get("chars") # 改為接收 'chars'

                if is_enabled:
                    yolo_logic.new_challenge_string = string_to_practice
                    yolo_logic.stop_challenge_mode = False
                    print(f"Challenge mode enabled for string: {string_to_practice}")
                else:
                    yolo_logic.stop_challenge_mode = True
                    yolo_logic.new_challenge_string = None
                    print("Challenge mode disabled.")

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
# 這會將 'frontend/build' 目錄下的檔案作為靜態資源提供
# 並將所有未匹配的路由導向到 index.html (SPA 模式)
app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
