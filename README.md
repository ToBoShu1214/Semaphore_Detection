# 智慧旗語教練系統 (Smart Semaphore Coach) 🚩

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://ultralytics.com/)

本專案是一個基於 **電腦視覺 (Computer Vision)** 與 **姿勢估計 (Pose Estimation)** 技術開發的即時旗語教學與偵測系統。與傳統基於影像分類的專案不同，本系統透過精確計算手臂角度提供**量化的教學矯正建議**，旨在成為旗語學習者的智慧虛擬教練。

---

## 🌟 核心特色 (Core Features)

### 1. 角度數據分析教學 (Objective Angle Guidance)
不同於「像不像」的模糊判斷，本系統直接分析人體關鍵點並計算雙臂角度。當姿勢不標準時，系統會給出精確的修正建議（如：「左手請抬高 15°」），實現真正的教育功能。

### 2. 雙旗語系統支援
*   **童軍旗語 (中文)**：完整支援中文數字序列與文字映射。
*   **國際旗語 (英文)**：全面整合 A-Z 字母與 0-9 數字，支援複雜的英數混合練習。

### 3. 三大練習子模式
*   **自由練習**：即時辨識使用者的動作並轉譯為對應文字。
*   **指定練習**：跟隨指定字串練習，具備錯誤判定與「取消」手勢回退機制。
*   **教學練習 (核心)**：專為初學者設計，系統會針對目標字元提供全程的角度指引，姿勢正確後自動跳轉。

### 4. 智慧鎖定與雙模型協同
*   **YOLO-Pose**：進行高精度的 2D 人體姿態追蹤。
*   **旗幟偵測**：透過自訓練的旗幟模型，自動在多人場景中鎖定「旗手」，並支援免旗幟 fallback 模式。

### 5. 高效能即時通訊
利用 **WebSocket** 實現後端 OpenCV 處理幀與前端 React 介面的低延遲同步，並支援 GPU 加速運算。

---

## 🛠️ 技術架構 (Architecture)

*   **Frontend**: React (TypeScript), CSS3 (Flexbox/Grid), WebSocket API.
*   **Backend**: Python FastAPI, Uvicorn.
*   **AI Engine**: Ultralytics YOLOv11-Pose, Custom Flag Detection Model.
*   **Libraries**: OpenCV, NumPy, Pillow.

---

## 🚀 快速開始 (Quick Start)

### 1. 環境需求
*   Python 3.9+
*   Node.js & npm
*   (建議) NVIDIA GPU 搭配 CUDA 環境

### 2. 後端啟動
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 3. 前端啟動
```bash
cd frontend
npm install
npm start
```

---

## 📸 介面預覽與操作
1.  **啟動手勢**：雙手舉起交叉以進入系統。
2.  **就位姿勢**：雙手垂直放於身體兩側。
3.  **取消手勢**：左手 45°, 右手 135°，用於刪除錯誤輸入。
4.  **鏡像功能**：支援畫面與圖示同步水平翻轉，方便對鏡練習。

---

## 🎓 專案價值
本專案克服了傳統深度學習分類模型無法解釋「為什麼錯」的缺點，將旗語教學科學化、量化，為視訊輔助教學提供了全新的解決方案。
