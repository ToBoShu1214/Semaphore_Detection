# 🚩 旗語辨識教學系統 (Semaphore Recognition & Teaching System)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)

本專案為一套基於電腦視覺 (Computer Vision) 與姿態估計 (Pose Estimation) 技術開發的即時旗語學習與辨識平台。系統透過精確計算手臂幾何夾角與骨架拉伸比，提供量化的教學建議，旨在輔助學習者精準掌握標準旗語動作。

---

## 🌟 系統特色 (System Features)

### 1. 量化教學與嚴格判定 (Quantitative Guidance)
系統實時分析人體關鍵點，並計算雙臂 360° 空間夾角與 180° 手肘平整度。
*   **高標判定**：手臂伸直判定採用 170° 夾角門檻與 0.9 骨架拉伸比，確保動作之標準度。
*   **動態校正**：在練習模式下，系統會提供即時的「抬/降」角度修正建議（如「抬 15°」），實現精準的數據驅動教學。

### 2. 雙重角色與多元模式
*   **揮旗手 (Sender)**：
    *   **自由練習**：即時辨識動作序列並轉譯為字元。
    *   **指定練習**：針對目標字串挑戰，具備嚴格的錯誤鎖定與取消手勢重置機制。
    *   **隨機測驗**：隨機抽取 5 題不重複題目，採 `正確次數 / 總嘗試次數` 的嚴格命中率計分。
*   **觀察員 (Receiver)**：
    *   **基礎教學**：透過字卡循序漸進學習信號辨識。
    *   **隨機測驗**：播放旗語動作，由使用者進行 10 題隨機不重複的選擇題填字測驗。

### 3. 智慧判定與衝突優化
*   **體系支援**：完整支持童軍旗語 (中文) 與海軍旗語 (英文)。
*   **英數自動切換**：針對海軍旗語實作了智慧切換邏輯。系統會自動偵測上下文，導引使用者比出「數字號 (#)」或「字母號 (J)」，並於切換後自動更新目標。
*   **權重鎖定邏輯**：導入「握旗狀態」、「中心距離」與「人體面積」之多維度權重評分，確保在多人環境下精準鎖定操作者。

### 4. 效能監控與硬體適應
*   **實時監測**：視訊畫面即時顯示 **Backend FPS**、**單幀處理延遲 (ms)** 與 **運算單元 (CUDA/CPU)**。
*   **自動優化**：支援 NVIDIA GPU 加速與 FP16 運算，無顯卡環境則自動切換至 CPU，確保相容性。
*   **攝影機持久化**：修正了切換模式時影像重設的問題，確保攝影機選擇在操作過程中始終保持穩定。

### 5. 靈活配置與診斷
*   **外部配置優先**：打包後可直接透過外部 `config.json` 調整判定閾值與控制台開關，無需重新編譯。
*   **自動化診斷**：內建日誌系統與連線狀態自動偵測，確保在各類網路與系統環境下的穩定啟動。

---

## 🛠️ 技術架構 (Architecture)

*   **前端介面**: React 18, TypeScript, WebSocket, Canvas API.
*   **後端引擎**: Python 3.11, FastAPI, YOLOv11-Pose (關鍵點擷取), YOLO (旗幟偵測), OpenCV.
*   **產品封裝**: PyInstaller, pywebview 原生視窗化技術。
*   **反饋系統**: 四層級同步音效系統與成果放大動畫特效。

---

## 🚀 快速開始 (Quick Start)

### 1. 直接運行 (正式版 APP)
若您無需修改原始碼，建議直接下載預先封裝好的執行檔，**免安裝 Python 環境**：
- **下載連結**：[👉 前往最新 Release 頁面](https://github.com/ToBoShu1214/Semaphore_Detection/releases/latest)
- **說明**：下載所有分卷 (`.001`, `.002`) 後放置於同一目錄，雙擊執行 **`旗語辨識教學系統.exe`** 即可。

### 2. 開發者環境配置
若需進行二次開發，請依照以下步驟配置：

#### A. 後端配置 (Backend)
```bash
# 安裝必要依賴項
cd backend
python -m pip install -r requirements.txt

# 啟動服務
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
本專案成功將旗語學習從主觀模仿轉化為客觀數據訓練，解決了傳統教學中動作標準模糊、缺乏即時回饋的痛點，為視訊輔助教學 (Video-based Coaching) 提供了一個高效且具備高度擴展性的解決方案。

---
**[詳細技術細節與全開發里程碑請參閱 backend/GEMINI.md]**
