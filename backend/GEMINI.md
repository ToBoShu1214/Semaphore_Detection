# 核心目標：教育與矯正 (Core Goal: Education and Correction)
本專案的核心目標是打造一個**具備教育功能的智慧旗語教練**。

與傳統僅能「辨識對錯」的影像分類模型不同，本專案的重點在於**「為什麼錯」**和**「如何修正」**。透過 `YOLO-Pose` 模型偵測人體姿態，系統能即時計算出使用者雙臂的**精確角度**。當姿勢不標準時，系統不再只給出「錯誤」的籠統結果，而是能提供如「左抬 15°」、「右降 10°」等**具體、可量化的修正建議**。

這種以**角度數據**為基礎的客觀回饋，是本專案實現教育目的的基石。

## ⚙️ 主要元件說明
| 元件 | 功能描述 |
| :--- | :--- |
| `main.py` | **後端伺服器 (FastAPI)**：處理 WebSocket 通訊，負責接收前端指令與傳送偵測結果。 |
| `yolo_logic.py` | **核心偵測引擎**：包含雙模型協同（YOLO11-Pose + Flag Detection）、狀態機邏輯與角度校正演算法。 |
| `VideoStream.tsx` | **前端介面 (React)**：負責視訊串流渲染、模式切換、資料視覺化與動作參考圖示顯示。 |

## 🤖 核心邏輯：狀態機 (最新版)
1.  **`IDLE` (閒置)**：提示「舉起雙手交叉啟動」。
2.  **`WAITING` (準備)**：提示「雙手放下預備」。
3.  **`READY` (就緒)**：系統待命，可開始比劃。
4.  **`DETECTING` (穩定中)**：偵測到初步姿勢，確認是否穩定維持。
5.  **`GRACE_PERIOD` (判定中)**：手臂伸直與角度判定。
6.  **`COOLDOWN` (冷卻)**：成功後強制放下雙手，才能進行下一個信號。

---

## 🚀 執行指南 (Installation & Setup)

### 一、環境需求
*   Python 3.9 ~ 3.11
*   Node.js & npm
*   (推薦) NVIDIA GPU 以獲得最佳流暢度 (FPS)

### 二、後端安裝 (CPU 模式)
```bash
# 建立環境
conda create -n semaphore python=3.11 -y
conda activate semaphore

# 安裝依賴
pip install fastapi uvicorn websockets numpy opencv-python ultralytics Pillow python-multipart torch torchvision
```

### 三、後端安裝 (GPU 模式)
請根據您的 CUDA 版本安裝對應的 PyTorch。以下為 CUDA 12.1 範例：
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn websockets numpy opencv-python ultralytics Pillow python-multipart
```

### 四、啟動步驟
1.  **啟動後端**: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
2.  **啟動前端**: `cd frontend && npm start`

---

### 五、開發進度日誌
| 日期 | 進度 |
| :--- | :--- |
| 2026/03/15 | **「教學練習模式」與系統深度整合**<br>- **新增教學模式**: 實作量化教學建議（抬/降 X°），不判定錯誤鎖定，專供學習。<br>- **整合國際旗語**: 國際旗語系統支援英數混合（A-Z, 0-9），自動對應信號。<br>- **效能大幅提升**: 旗幟偵測改為 5 幀/次，靈敏度放寬至 150px。<br>- **UI 穩定化**: 提示區固定高度 (160px)，支援圖示鏡像水平翻轉。<br>- **Bug 修正**: 解決 NumPy 與 JSON 序列化導致的後端崩潰。 |
| 2025/12/08 | **「指定練習」修正與 GPU 相容性優化**<br>- 修正指定練習結束後的手勢 Bug。<br>- 升級模型至 `yolo11s-pose.pt` 與 `flag.pt`。 |

---

## 📝 未來代辦清單 (To-Do List)
- [ ] **完善考試模式 (Exam Mode)**：目前後端尚缺乏完整的計分與評分邏輯，僅具備基礎判別。
- [ ] **多語言支援**：將前端 UI 字串抽離至 i18n 檔案，支援英文語系介面。
- [ ] **音效回饋**：當姿勢正確 (OK) 或偵測成功時，加入提示音效，增強互動感。
- [ ] **練習數據導出**：紀錄使用者的平均角度誤差，產出練習報告圖表。
- [ ] **多人比拼模式**：支援同時追蹤兩位旗手，進行速度比賽。
- [ ] **手勢靈敏度動態調整**：允許使用者在前端自行調整 `STABLE_DELAY` 與容錯角度。
