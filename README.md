# 🚩 旗語辨識教學系統 (Semaphore Recognition & Teaching System)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)

本專案為一套基於電腦視覺 (Computer Vision) 與姿態估計 (Pose Estimation) 技術開發的即時旗語學習與辨識平台。系統透過精確計算手臂幾何夾角，提供量化的教學建議，旨在輔助學習者精準掌握旗語動作。

---

## 🌟 系統特色 (System Features)

### 1. 量化教學引導 (Quantitative Guidance)
系統實時分析人體關鍵點，並計算雙臂相對於身體的 360° 空間夾角。在教學模式下，系統提供即時的「抬/降」角度修正建議，實現數據驅動的互動反饋。

### 2. 多元旗語體系支援
*   **童軍旗語 (中文)**：完整支援 4 位數信號序列與中文字元映射。
*   **國際旗語 (英文)**：整合 A-Z 字母與 0-9 數字定義，支援英數混合練習。

### 3. 三大練習模式
*   **自由練習**：實時辨識並轉譯使用者之旗語動作。
*   **指定練習**：針對特定字串進行挑戰，具備錯誤鎖定與重置機制。
*   **教學引導 (核心)**：無視誤判限制，全程以角度偏差指令導引使用者達成標準動作。

---

## 🛠️ 技術架構 (Architecture)

*   **前端介面**: React, TypeScript, WebSocket, Canvas API.
*   **後端引擎**: Python, FastAPI, YOLOv11 (Pose & Detection), OpenCV.
*   **產品封裝**: PyInstaller, pywebview.
*   **回饋系統**: 四層級即時音效反饋與動畫特效。

---

## 🚀 快速開始 (Quick Start)

### 1. 直接運行 (正式版 APP)
若您無需修改程式碼，建議直接下載預先封裝好的執行檔，**免安裝 Python 環境**：
- **下載連結**：[👉 前往最新 Release 頁面](https://github.com/ToBoShu1214/Semaphore_Detection/releases/latest)
- **說明**：下載所有分卷 (`.001`, `.002`) 後放置於同一目錄，使用 7-Zip 或 WinRAR 對第一個檔案解壓，執行 **`旗語辨識教學系統.exe`** 即可。

### 2. 開發者環境配置
若需進行開發，請依照以下步驟配置並啟動：

#### A. 後端配置 (Backend)
```bash
# 建立並啟用 Conda 環境
conda create -n semaphore python=3.11 -y
conda activate semaphore

# 安裝 PyTorch (以 CUDA 12.4 為例)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安裝必要依賴項 (包含 UI 引擎與封裝工具)
python -m pip install fastapi uvicorn websockets numpy opencv-python ultralytics Pillow python-multipart lapx pywebview pyinstaller

# 啟動後端伺服器
cd backend
python main.py
```

#### B. 前端啟動 (Frontend)
```bash
# 安裝依賴項並啟動開發伺服器
cd frontend
npm install
npm start
```

#### C. 使用啟動器 (Launcher)
```bash
# 自動開啟後端並彈出原生視窗介面
cd backend
python launcher.py
```

---

## 🎓 專案價值
本專案成功將旗語學習從主觀模仿轉化為客觀數據訓練，為視訊輔助教學 (Video-based Coaching) 提供了一個高效、穩定且具備高度擴展性的解決方案。

---
**[技術細節與開發日誌詳見 backend/GEMINI.md]**
