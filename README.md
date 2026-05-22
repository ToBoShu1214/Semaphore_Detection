# 🚩 旗語教學系統 (Semaphore Teaching System)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)

本專案為一套基於電腦視覺 (Computer Vision) 與姿態估計 (Pose Estimation) 技術開發的即時旗語學習與辨識平台。系統透過精確計算手臂幾何夾角，提供量化的教學建議，旨在輔助學習者精準掌握標準旗語動作。

---

## 🌟 系統特色 (System Features)

### 1. 量化教學引導 (Quantitative Guidance)
系統實時分析人體關鍵點，並計算雙臂相對於身體中心線的 360° 空間夾角。在教學模式下，系統會針對目標信號提供即時的「抬/降」角度修正建議與側別判定，實現數據驅動的互動引導。

### 2. 多元旗語體系與智慧判定
*   **體系支援**：完整支持童軍旗語 (中文) 4 位數映射與國際旗語 (英文) 英數定義。
*   **模式切換引導**：針對國際旗語實作了英數切換邏輯。系統能自動辨識上下文，在需要時導引使用者比出「數字號 (#)」或「字母號 (J)」，並於切換完成後自動更新下一個學習目標。
*   **衝突優化**：針對動作重疊的字元（如 A 與 1），系統具備「字元優先級」判定邏輯，能根據當前練習目標精準辨識意圖，解決狀態跳針問題。

### 3. 三大練習模式
*   **自由練習**：實時辨識並轉譯使用者之旗語動作序列。
*   **指定練習**：針對特定目標字串進行挑戰，具備嚴格的錯誤鎖定與取消手勢重置機制。
*   **教學引導**：無視誤判限制，全程以角度偏差數據導引使用者達成標準動作。

 ### ⚠️ 注意事項 (WIP)
 - **考試模式 (Exam Mode)**: 目前尚在開發階段，相關評分與題庫邏輯尚未完備，本版本目前僅供練習與教學功能測試。

### 4. 自動化硬體適應
*   系統具備動態設備偵測能力，支援 NVIDIA GPU (CUDA) 加速與 FP16 半精度運算；若無獨立顯卡，系統將自動切換至 CPU 模式運行，確保在不同性能硬體上的相容性。

### 5. 連線穩定性與自動化診斷
*   **動態主機偵測**：前端 WebSocket 具備 Host 自動偵測功能，能根據執行環境（Localhost 或局域網 IP）自動握手，解決封裝後在特定網路環境下的連線失效問題。
*   **啟動日誌系統**：內建 `backend_error.log` 自動日誌機制。若系統在特殊限制環境（如校園網域）啟動失敗，使用者可透過日誌快速定位問題原因。

---

## 🛠️ 技術架構 (Architecture)

*   **前端介面**: React, TypeScript, WebSocket, Canvas API.
*   **後端引擎**: Python, FastAPI, YOLOv11 (Pose & Detection), OpenCV.
*   **產品封裝**: PyInstaller, pywebview 原生視窗化技術。
*   **反饋系統**: 四層級同步音效系統與成果放大動畫特效。

---

## 🚀 快速開始 (Quick Start)

### 1. 直接運行 (正式版 APP)
若您無需修改原始碼，建議直接下載預先封裝好的執行檔，**免安裝 Python 環境**：
- **下載連結**：[👉 前往最新 Release 頁面](https://github.com/ToBoShu1214/Semaphore_Detection/releases/latest)
- **說明**：下載所有分卷 (`.001`, `.002`) 後放置於同一目錄，使用 7-Zip 或 WinRAR 對第一個檔案解壓，雙擊執行 **`旗語教學系統.exe`** 即可。
- **注意**：首次啟動時，系統載入 AI 模型需要約 5-15 秒（視電腦性能而定），介面跳出後即可開始使用。

### 2. 開發者環境配置
若需進行二次開發，請依照以下步驟配置並啟動：

#### A. 後端配置 (Backend)
```bash
# 建立並啟用環境
conda create -n semaphore python=3.11 -y
conda activate semaphore

# 安裝 PyTorch (以 CUDA 12.4 為例)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安裝必要依賴項
python -m pip install fastapi uvicorn websockets numpy opencv-python ultralytics Pillow python-multipart lapx pywebview pyinstaller

# 啟動服務
cd backend
python main.py
```

#### B. 前端啟動 (Frontend)
```bash
cd frontend
npm install
npm run build
```

---

## 🎓 專案價值
本專案成功將旗語學習從主觀模仿轉化為客觀數據訓練，為視訊輔助教學 (Video-based Coaching) 提供了一個高效、穩定且具備高度擴展性的解決方案。

---
**[詳細技術細節與全開發里程碑請參閱 backend/GEMINI.md]**
