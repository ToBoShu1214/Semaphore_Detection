# 🚩 旗語辨識教學系統 (Semaphore Recognition & Teaching System)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)

本專案是一個基於 **電腦視覺 (Computer Vision)** 與 **姿態估計 (Pose Estimation)** 技術開發的即時旗語學習與辨識平台。與傳統僅能判斷姿勢「像不像」的分類模型不同，本系統透過精確計算手臂幾何角度，提供**量化的教學校正建議**，旨在成為旗語學習者的專業虛擬教練。

---

## 🌟 核心特色 (Core Features)

### 1. 量化教學引導 (Objective Guidance)
系統直接分析人體 17 個關鍵點並換算為雙臂相對於身體的 360° 夾角。在「教學模式」下，系統會根據目標信號提供即時的「抬/降」角度修正建議（例如：「左抬 15°」），實現真正具備教育意義的反饋。

### 2. 多元旗語系統整合
*   **童軍旗語 (中文)**：全面支援 4 位數序列與中文漢字的映射。
*   **國際旗語 (英文)**：完美整合 A-Z 字母與 0-9 數字，支援英數混合字串練習。

### 3. 三大練習模式
*   **自由練習**：隨意揮劃，系統即時轉譯為文字。
*   **指定練習**：針對目標字串挑戰，具備嚴格判定與取消手勢重置機制。
*   **教學練習 (核心)**：無視誤判，全程以角度指令引導使用者達成標準姿勢。

### 4. 智慧鎖定與雙模型技術
利用 **YOLO11-Pose** 進行動態追蹤，並結合自定義的 **Flag Detection** 模型。在多人環境下，系統能透過計算手腕與旗幟的距離，精準鎖定正在動作的旗手。

---

## 🛠️ 技術架構 (Architecture)

*   **Frontend**: React, TypeScript, CSS Grid/Flexbox, WebSocket.
*   **Backend**: Python, FastAPI, Uvicorn.
*   **AI Engine**: YOLOv11 (Pose & Detection), OpenCV.
*   **Audio System**: 實作四層級即時音效反饋，增強互動節奏感。

---

## 📈 專案演進歷史 (Project Evolution)

本專案經歷了多個技術迭代階段：
*   **Phase 1 (2025/09)**：初期實驗，使用 YOLOv8-Pose 確立角度計算可行性。
*   **Phase 2 (2025/11)**：架構 Web 化，導入 WebSocket 實現低延遲視訊傳輸。
*   **Phase 3 (2025/12)**：模型升級至 YOLO11 世代，強化 GPU 相容性。
*   **Final Phase (2026/03)**：全面優化量化教學邏輯、加入多層級音效與成果放大動畫，達成商業等級的互動體驗。

---

## 🚀 快速開始 (Quick Start)

### 1. 環境準備
請確保已安裝 **Conda**、**Python 3.11** 與 **Node.js**。建議配備 NVIDIA GPU 以提升辨識流暢度。

### 2. 後端啟動 (Backend)
```bash
cd backend
conda create -n semaphore python=3.11 -y
conda activate semaphore
pip install -r requirements.txt
python main.py
```

### 3. 前端啟動 (Frontend)
```bash
cd frontend
npm install
npm start
```

---

## 📸 操作指南
1.  **啟動系統**：舉起雙手交叉（Cross Gesture）。
2.  **就位準備**：雙手垂直放下（Stay Gesture）。
3.  **動作採計**：姿勢穩定 0.4 秒且手臂伸直後，系統自動確認。
4.  **取消錯誤**：比出取消手勢（左 45°, 右 135°）可回退最後一步。

---

## 🎓 專案價值
本專案成功將「旗語學習」從主觀的模仿轉化為客觀的數據訓練，為視訊輔助教學（Video-based Coaching）提供了一個高效、穩定且具備高度擴展性的解決方案。
