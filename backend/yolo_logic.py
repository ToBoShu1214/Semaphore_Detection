import os
import sys

# Define the path to the CUDA bin directory
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"

# Proactively add the CUDA bin directory to the DLL search path
# This is a more robust way to ensure DLLs are found on Windows
# See: https://docs.python.org/3/library/os.html#os.add_dll_directory
if sys.version_info >= (3, 8) and sys.platform == "win32":
    try:
        if os.path.exists(cuda_bin_path):
            os.add_dll_directory(cuda_bin_path)
            print(f"[INFO] Added {cuda_bin_path} to DLL search path.")
        else:
            print(f"[WARNING] CUDA bin path not found: {cuda_bin_path}")
    except Exception as e:
        print(f"[WARNING] Could not add {cuda_bin_path} to DLL search path. Error: {e}")
        pass

import cv2
import math
import time
import csv
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
from collections import deque
import os
import argparse
import json
import onnxruntime

# 用於在 main.py 和 yolo_logic.py 之間通訊的模組級變數
new_challenge_string = None
stop_challenge_mode = False

def create_video_capture(video_source_str):
    """從字串來源建立視訊捕捉物件。"""
    try:
        video_source = int(video_source_str)
    except ValueError:
        video_source = video_source_str
    return cv2.VideoCapture(video_source)

def run_detection(video_source_str='0', model_path='yolov8n-pose.pt', flag_model_path='flag.onnx', mapping_csv_path='mapping.csv', current_mode='practice', target_sequence=None, start_exam_signal=False, stop_exam_signal=False):
    # -----------------------
    # 載入設定檔 (Load Configuration)
    # -----------------------
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("錯誤：找不到 config.json 設定檔。請確保它與 main.py 在同一目錄下。")
        return
    except json.JSONDecodeError:
        print("錯誤：config.json 格式錯誤。請檢查 JSON 語法。")
        return

    # 從設定檔讀取常數
    # 主要控制常數
    ANGLE_TOL_STRICT = config['ANGLE_TOL_STRICT']
    ANGLE_TOL_NORMAL = config['ANGLE_TOL_NORMAL']
    STRAIGHT_ARM_THRESHOLD = config['STRAIGHT_ARM_THRESHOLD']
    STABLE_DELAY = config['STABLE_DELAY']
    READY_ANGLE = config['READY_ANGLE']
    STRAIGHTEN_GRACE_PERIOD = config['STRAIGHTEN_GRACE_PERIOD']
    SMOOTHING_WINDOW_SIZE = config['SMOOTHING_WINDOW_SIZE']
    PROMPT_DELAY_AFTER_SUCCESS = config['PROMPT_DELAY_AFTER_SUCCESS']

    # 手勢偵測常數
    GESTURE_ANGLE_THRESHOLD = config['GESTURE_ANGLE_THRESHOLD']
    GESTURE_UPWARD_STRAIGHT_ARM_THRESHOLD = config['GESTURE_UPWARD_STRAIGHT_ARM_THRESHOLD']
    GESTURE_CROSS_COUNT_THRESHOLD = config['GESTURE_CROSS_COUNT_THRESHOLD']
    GESTURE_TIMEOUT = config['GESTURE_TIMEOUT']
    GESTURE_WRIST_DISTANCE_THRESHOLD = config['GESTURE_WRIST_DISTANCE_THRESHOLD']
    GESTURE_CROSS_BUFFER = config['GESTURE_CROSS_BUFFER']

    # 目標偵測常數
    FLAG_DETECTION_INTERVAL = config['FLAG_DETECTION_INTERVAL']
    MIN_IOU_THRESHOLD = config['MIN_IOU_THRESHOLD']
    TARGET_LOST_TIMEOUT = config['TARGET_LOST_TIMEOUT']

    # 顯示設定
    DISPLAY_WIDTH = config['DISPLAY_WIDTH']

    # -----------------------
    # 數字角度設定
    # -----------------------
    # This dictionary depends on the constants loaded from the config file.
    number_angles = {
        0: ((180, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)), 1: ((READY_ANGLE, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)),
        2: ((READY_ANGLE, ANGLE_TOL_STRICT), (90, ANGLE_TOL_STRICT)), 3: ((READY_ANGLE, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)),
        4: ((READY_ANGLE, ANGLE_TOL_STRICT), (180, ANGLE_TOL_STRICT)), 5: ((135, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)),
        6: ((90, ANGLE_TOL_STRICT), (READY_ANGLE, ANGLE_TOL_STRICT)), 7: ((45, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)),
        8: ((315, ANGLE_TOL_NORMAL), (90, ANGLE_TOL_STRICT)), 9: ((300, ANGLE_TOL_NORMAL), (135, ANGLE_TOL_NORMAL))
    }

    # -----------------------
    # 輔助函式
    # -----------------------
    def angle_diff(angle1, angle2):
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def calc_angle_360(p1, p2, p3, hand='left'):
        a_x, a_y = p1[0] - p2[0], p1[1] - p2[1]
        b_x, b_y = p3[0] - p2[0], p3[1] - p2[1]
        dot, det = a_x * b_x + a_y * b_y, a_x * b_y - a_y * b_x
        angle = math.degrees(math.atan2(det, dot))
        if hand == 'right': angle = -angle
        return (angle + 360) % 360

    def calc_angle_180(p1, p2, p3):
        if any(coord < 1 for point in [p1, p2, p3] for coord in point): return 0.0
        try:
            v1, v2 = (p1[0] - p2[0], p1[1] - p2[1]), (p3[0] - p2[0], p3[1] - p2[1])
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag_v1, mag_v2 = math.sqrt(v1[0]**2 + v1[1]**2), math.sqrt(v2[0]**2 + v2[1]**2)
            if mag_v1 == 0 or mag_v2 == 0: return 180.0
            cosine_angle = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
            return math.degrees(math.acos(cosine_angle))
        except (ValueError, ZeroDivisionError): return 180.0

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    def recognize_pose(l_angle, r_angle):
        if angle_diff(l_angle, 45) <= ANGLE_TOL_NORMAL and angle_diff(r_angle, 135) <= ANGLE_TOL_NORMAL: return "cancel"
        for num, ((l_target, l_tol), (r_target, r_tol)) in number_angles.items():
            if angle_diff(l_angle, l_target) <= l_tol and angle_diff(r_angle, r_target) <= r_tol: return num
        return None

    def is_ready_pose(l_angle, r_angle):
        return angle_diff(l_angle, READY_ANGLE) <= ANGLE_TOL_STRICT and angle_diff(r_angle, READY_ANGLE) <= ANGLE_TOL_STRICT

    def is_hands_above_head(kpts):
        if not kpts.any() or len(kpts) < 11: return False
        nose, l_wrist, r_wrist = kpts[0], kpts[9], kpts[10]
        if nose[1] > 0 and l_wrist[1] > 0 and r_wrist[1] > 0:
            return l_wrist[1] < nose[1] and r_wrist[1] < nose[1]
        return False

    def load_mapping(csv_file):
        mapping, reverse_mapping = {}, {}
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        digit_seq, char = row[0].strip(), row[1].strip()
                        mapping[digit_seq] = char
                        reverse_mapping[char] = digit_seq
        except FileNotFoundError: print(f"警告：找不到對應表檔案 {csv_file}。")
        return mapping, reverse_mapping

    # -----------------------
    # 初始化
    # -----------------------
    # For the YOLO model, the ultralytics library automatically detects and uses an available GPU.
    # We just need to ensure the environment (e.g., from requirements-gpu.txt) is set up correctly.
    pose_model = YOLO(model_path)
    try:
        # First, try to initialize with CUDA provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        flag_model = onnxruntime.InferenceSession(flag_model_path, providers=providers)
    except Exception as e:
        # If CUDA fails, print a warning and try again with only CPU
        print(f"[WARNING] Failed to initialize ONNX with CUDAExecutionProvider. Error: {e}")
        print("[INFO] Falling back to CPUExecutionProvider.")
        try:
            providers = ['CPUExecutionProvider']
            flag_model = onnxruntime.InferenceSession(flag_model_path, providers=providers)
        except Exception as e_cpu:
            # If even CPU fails, then it's a fatal error
            print(f"錯誤：無法使用 CPU 載入 ONNX 旗幟模型 '{flag_model_path}'. 錯誤: {e_cpu}")
            return

    # Get the actual provider being used to give accurate feedback to the user.
    actual_provider = flag_model.get_providers()[0]
    print(f"[INFO] ONNX model is running on: {actual_provider}")

    mapping, reverse_mapping = load_mapping(mapping_csv_path)
    cap = create_video_capture(video_source_str)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影像來源 '{video_source_str}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = frame_width / frame_height if frame_height > 0 else 16/9
    DISPLAY_HEIGHT = int(DISPLAY_WIDTH / aspect_ratio)

    # -- 狀態機與控制變數 --
    state, state_timer, current_digit, sequence, display_result, result_time = "IDLE", 0, None, [], None, 0
    word_history, last_completed_sequence = [], []
    
    # -- 目標追蹤變數 --
    target_person_id = None
    frames_since_flag_detection = FLAG_DETECTION_INTERVAL
    last_known_target_kpts = np.array([])
    last_known_target_person_bbox = None
    target_lost_start_time = 0.0

    # -- 手勢偵測變數 --
    cross_sub_state, cross_count, last_gesture_time, gesture_complete = "UNCROSSED", 0, 0, False

    # -- 角度平滑化佇列 --
    history = {k: deque(maxlen=SMOOTHING_WINDOW_SIZE) for k in ['left_angle', 'right_angle', 'left_elbow', 'right_elbow']}

    # =======================
    #   主迴圈
    # =======================
    while True:
        current_time = time.time()
        is_challenge_mode = current_mode == 'challenge'
        global new_challenge_string, stop_challenge_mode
        if new_challenge_string:
            challenge_target_string = new_challenge_string
            word_history = []
            challenge_current_word_index = 0
            next_char = challenge_target_string[challenge_current_word_index]
            current_char_target_sequence = list(reverse_mapping.get(next_char, ''))
            current_char_next_digit_index = 0
            new_challenge_string = None
        if stop_challenge_mode:
            is_challenge_mode = False
            current_mode = 'practice'
            stop_challenge_mode = False

        ret, frame = cap.read()
        if not ret:
            print("影片結束或讀取錯誤。")
            break

        detection_data = {
            "state": state, "prompt_code": None, "cross_count": cross_count,
            "left_angle": None, "right_angle": None, "current_digit": None,
            "sequence": list(sequence), "display_result": display_result,
            "target_person_bbox": None, "flag_boxes": [], "mode": current_mode,
            "word_history": list(word_history),
        }

        pose_results = pose_model.track(frame.copy(), persist=True, verbose=False, show=False)
        
        all_person_boxes = {}
        all_person_kpts = {}
        if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None:
            for box, kpts in zip(pose_results[0].boxes, pose_results[0].keypoints):
                if box.id is not None:
                    person_id = int(box.id.item())
                    all_person_boxes[person_id] = box.xyxy[0].cpu().numpy()
                    all_person_kpts[person_id] = kpts.xy[0].cpu().numpy()

        frames_since_flag_detection += 1
        if target_person_id is None:
            frames_since_flag_detection = 0
            img_resized = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)
            onnx_inputs = {flag_model.get_inputs()[0].name: img_tensor}
            onnx_outputs = flag_model.run(None, onnx_inputs)
            output_data = onnx_outputs[0][0]
            candidates = output_data.T
            conf_threshold = 0.5
            scores_raw = candidates[:, 4]
            good_candidates_mask = scores_raw > conf_threshold
            candidates = candidates[good_candidates_mask]
            scores = scores_raw[good_candidates_mask]
            boxes_cxcywh = candidates[:, :4]
            boxes_xywh = boxes_cxcywh.copy()
            boxes_xywh[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
            boxes_xywh[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
            iou_threshold = 0.5
            selected_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)
            flag_boxes = []
            if len(selected_indices) > 0:
                for i in selected_indices.flatten():
                    x_center, y_center, width, height = boxes_cxcywh[i]
                    x1 = int((x_center - width / 2) * frame_width / 640)
                    y1 = int((y_center - height / 2) * frame_height / 640)
                    x2 = int((x_center + width / 2) * frame_width / 640)
                    y2 = int((y_center + height / 2) * frame_height / 640)
                    flag_boxes.append([x1, y1, x2, y2])
            detection_data["flag_boxes"] = flag_boxes
            if flag_boxes and all_person_kpts:
                best_person_id, max_iou = -1, -1
                for person_id, kpts in all_person_kpts.items():
                    l_wrist, r_wrist = kpts[9], kpts[10]
                    if l_wrist[0] > 0 and l_wrist[1] > 0 and r_wrist[0] > 0 and r_wrist[1] > 0:
                        padding = 20
                        x1, y1 = int(min(l_wrist[0], r_wrist[0]) - padding), int(min(l_wrist[1], r_wrist[1]) - padding)
                        x2, y2 = int(max(l_wrist[0], r_wrist[0]) + padding), int(max(l_wrist[1], r_wrist[1]) + padding)
                        hands_box = [x1, y1, x2, y2]
                        current_max_iou = 0
                        for flag_box in flag_boxes:
                            current_max_iou = max(current_max_iou, iou(hands_box, flag_box))
                        if current_max_iou > max_iou:
                            max_iou, best_person_id = current_max_iou, person_id
                if max_iou > MIN_IOU_THRESHOLD:
                    target_person_id = best_person_id

        target_kpts = np.array([])
        target_detected_this_frame = False
        if target_person_id is not None and target_person_id in all_person_kpts:
            target_detected_this_frame = True
            current_frame_target_kpts = all_person_kpts[target_person_id]
            last_known_target_kpts = current_frame_target_kpts
            last_known_target_person_bbox = all_person_boxes[target_person_id].tolist()
            target_lost_start_time = 0.0
            target_kpts = current_frame_target_kpts
        else:
            if target_person_id is not None:
                if target_lost_start_time == 0.0:
                    target_lost_start_time = current_time
                if (current_time - target_lost_start_time) > TARGET_LOST_TIMEOUT:
                    target_person_id = None
                    last_known_target_kpts = np.array([])
                    last_known_target_person_bbox = None
                    for q in history.values(): q.clear()
                    word_history.clear()
                    sequence.clear()
                    target_lost_start_time = 0.0
                else:
                    target_kpts = last_known_target_kpts
            else:
                target_kpts = np.array([])
                last_known_target_kpts = np.array([])
                last_known_target_person_bbox = None
                target_lost_start_time = 0.0

        if target_kpts.any():
            history['left_angle'].append(calc_angle_360(target_kpts[9], target_kpts[5], target_kpts[11], hand='left'))
            history['right_angle'].append(calc_angle_360(target_kpts[10], target_kpts[6], target_kpts[12], hand='right'))
            history['left_elbow'].append(calc_angle_180(target_kpts[5], target_kpts[7], target_kpts[9]))
            history['right_elbow'].append(calc_angle_180(target_kpts[6], target_kpts[8], target_kpts[10]))
        else: 
            for q in history.values(): q.clear()

        if len(history['left_angle']) >= SMOOTHING_WINDOW_SIZE:
            angles = {k: sum(v) / len(v) for k, v in history.items()}
            detection_data.update({"left_angle": int(angles['left_angle']), "right_angle": int(angles['right_angle']), "current_digit": current_digit})
            hands_up = is_hands_above_head(target_kpts)
            if gesture_complete and not hands_up:
                action = "開始" if state == "IDLE" else "重置"
                if action == "開始": state = "WAITING"
                else:
                    state = "IDLE"
                    target_person_id = None
                    word_history.clear()
                    last_completed_sequence.clear()
                    for q in history.values(): q.clear()
                gesture_complete, cross_count, cross_sub_state = False, 0, "UNCROSSED"
                sequence, current_digit, display_result = [], None, None
            elif hands_up and not gesture_complete:
                if time.time() - last_gesture_time > GESTURE_TIMEOUT:
                    cross_count, cross_sub_state = 0, "UNCROSSED"
                left_wrist_x, right_wrist_x = target_kpts[9][0], target_kpts[10][0]
                is_crossed = abs(left_wrist_x - right_wrist_x) < GESTURE_WRIST_DISTANCE_THRESHOLD
                is_uncrossed = abs(left_wrist_x - right_wrist_x) > (GESTURE_WRIST_DISTANCE_THRESHOLD + GESTURE_CROSS_BUFFER)
                if cross_sub_state == "UNCROSSED":
                    if is_crossed:
                        cross_count += 1
                        cross_sub_state = "CROSSED"
                        last_gesture_time = time.time()
                        if cross_count >= GESTURE_CROSS_COUNT_THRESHOLD: gesture_complete = True
                elif cross_sub_state == "CROSSED":
                    if is_uncrossed:
                        cross_sub_state = "UNCROSSED"
                        last_gesture_time = time.time()
            elif not hands_up:
                if state not in ["WAITING", "READY", "DETECTING", "GRACE_PERIOD", "COOLDOWN"]:
                     state = "IDLE"
            if state not in ["IDLE", "GESTURE_DONE"]:
                if is_challenge_mode and challenge_target_string: pass
                else:
                    detected_pose = recognize_pose(angles['left_angle'], angles['right_angle'])
                    if current_mode == 'practice':
                        if state == "WAITING":
                            if is_ready_pose(angles['left_angle'], angles['right_angle']): state = "READY"
                        elif state == "READY":
                            if detected_pose is not None: state, current_digit, state_timer = "DETECTING", detected_pose, time.time()
                        elif state == "DETECTING":
                            if detected_pose != current_digit: state, current_digit = "READY", None
                            elif time.time() - state_timer >= STABLE_DELAY: state, state_timer = "GRACE_PERIOD", time.time()
                        elif state == "GRACE_PERIOD":
                            if detected_pose != current_digit: state, current_digit = "READY", None
                            else:
                                is_pose_valid = False
                                if current_digit == "cancel": is_pose_valid = True
                                else:
                                    left_elbow_angle, right_elbow_angle = angles['left_elbow'], angles['right_elbow']
                                    (l_target, _), (r_target, _) = number_angles.get(current_digit, ((None, 0), (None, 0)))
                                    l_arm_ok = (l_target == READY_ANGLE) or (left_elbow_angle >= STRAIGHT_ARM_THRESHOLD)
                                    r_arm_ok = (r_target == READY_ANGLE) or (right_elbow_angle >= STRAIGHT_ARM_THRESHOLD)
                                    if l_arm_ok and r_arm_ok: is_pose_valid = True
                                if is_pose_valid:
                                    if current_digit == "cancel":
                                        if len(sequence) > 0: sequence.pop()
                                        elif len(word_history) > 0:
                                            word_history.pop()
                                            if last_completed_sequence:
                                                sequence = last_completed_sequence[:-1]
                                                last_completed_sequence = []
                                    elif len(sequence) < 4: sequence.append(str(current_digit))
                                    state, state_timer, current_digit = "COOLDOWN", time.time(), None
                                elif time.time() - state_timer > STRAIGHTEN_GRACE_PERIOD: state, current_digit = "READY", None
                        elif state == "COOLDOWN":
                            if is_ready_pose(angles['left_angle'], angles['right_angle']): state = "READY"

        prompt_code = None
        if not all_person_kpts: prompt_code = "等待人出現..."
        elif target_person_id is None: prompt_code = "正在尋找目標 (旗手)..."
        elif not target_detected_this_frame and (time.time() - target_lost_start_time) < TARGET_LOST_TIMEOUT:
             prompt_code = f"目標丟失... ({int(TARGET_LOST_TIMEOUT - (time.time() - target_lost_start_time))}s)"
        elif len(history['left_angle']) < SMOOTHING_WINDOW_SIZE: prompt_code = "目標已鎖定，正在穩定..."
        else:
            hands_up = is_hands_above_head(target_kpts)
            if state == "IDLE":
                if hands_up:
                    if cross_sub_state == "UNCROSSED": prompt_code = f"請交叉手腕 ({cross_count}/{GESTURE_CROSS_COUNT_THRESHOLD})"
                    else: prompt_code = f"請打開雙手 ({cross_count}/{GESTURE_CROSS_COUNT_THRESHOLD})"
                else: prompt_code = "請舉起雙手以開始手勢"
            elif state == "WAITING": prompt_code = "請做出 [雙手放下] 的準備姿勢"
            elif state == "READY": prompt_code = "準備就緒，請開始動作"
            elif state == "DETECTING": prompt_code = f"檢測到指令 {current_digit}，請保持穩定..."
            elif state == "GRACE_PERIOD": prompt_code = "手臂未伸直，請重做"
            elif state == "COOLDOWN": prompt_code = "請將雙手垂直放下，以輸入下一個數字"
            if is_challenge_mode and challenge_target_string and state not in ["IDLE", "GESTURE_DONE"]:
                target_digit_str = current_char_target_sequence[current_char_next_digit_index]
                prompt_code = f"請比出數字 {target_digit_str}"
                if current_digit is not None and str(current_digit) == target_digit_str:
                    prompt_code = f"正確！偵測到數字 {current_digit}，請保持..."
        detection_data["prompt_code"] = prompt_code
        detection_data["state"] = state
        detection_data["cross_count"] = cross_count
        detection_data["word_history"] = list(word_history)
        if current_mode == 'practice':
            detection_data["sequence"] = list(sequence)
            if len(sequence) == 4 and display_result is None:
                key = "".join(sequence)
                display_result, result_time = mapping.get(key, "□"), time.time()
                word_history.append(display_result)
                last_completed_sequence = list(sequence)
                sequence = []
            if display_result and time.time() - result_time > 0.8: display_result = None
            detection_data["display_result"] = display_result
        
        if target_person_id is not None and last_known_target_person_bbox is not None:
            detection_data["target_person_bbox"] = last_known_target_person_bbox
        else:
            detection_data["target_person_bbox"] = None

        # --- 產生輸出 ---
        # 修正：計算縮放比例，以正確繪製邊界框
        x_scale = DISPLAY_WIDTH / frame_width if frame_width > 0 else 0
        y_scale = DISPLAY_HEIGHT / frame_height if frame_height > 0 else 0

        resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # 在畫面上繪製框
        if detection_data["target_person_bbox"]:
            x1, y1, x2, y2 = map(int, detection_data["target_person_bbox"])
            # 應用縮放比例
            x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
            y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(resized_frame, f"Target ID: {target_person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        for box in detection_data["flag_boxes"]:
            x1, y1, x2, y2 = map(int, box)
            # 應用縮放比例
            x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
            y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        _, jpeg = cv2.imencode('.jpg', resized_frame)
        yield jpeg.tobytes(), detection_data

    cap.release()
    print("Detection loop finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semaphore Pose Detection Logic Test')
    parser.add_argument('--video', type=str, default='0', help='Path to video or "0" for webcam.')
    args = parser.parse_args()
    
    # Dummy generator to test the main loop
    def dummy_generator():
        for jpeg_bytes, data in run_detection(video_source_str=args.video):
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # --- Add all the display logic from main.py here for testing ---
            font_path = 'msjh.ttc'
            font = ImageFont.truetype(font_path, 40)
            font_small = ImageFont.truetype(font_path, 24)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # Display prompt
            prompt = data.get("prompt_code", "")
            if prompt:
                draw.text((50, 50), prompt, font=font, fill=(255, 255, 0))

            # Display sequence
            sequence = data.get("sequence", [])
            seq_text = " ".join(sequence)
            draw.text((50, 120), f"序列: [ {seq_text} ]", font=font, fill=(255, 255, 255))

            # Display result
            result = data.get("display_result")
            if result:
                draw.text((50, 190), f"結果: {result}", font=font, fill=(0, 255, 0))

            # Display word history
            history = data.get("word_history", [])
            history_text = " ".join(history)
            draw.text((50, frame.shape[0] - 70), f"歷史: {history_text}", font=font_small, fill=(200, 200, 200))

            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow('Semaphore Detection Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    dummy_generator()
    cv2.destroyAllWindows()
