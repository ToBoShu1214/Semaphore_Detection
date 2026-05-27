import os
import sys
import cv2
import math
import time
import csv
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
from collections import deque
import argparse
import json
import torch

# 取得資源路徑 (PyInstaller 專用邏輯)
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
        
    return os.path.join(base_path, relative_path)

def create_video_capture(video_source_str):
    try:
        video_source = int(video_source_str)
    except ValueError:
        video_source = video_source_str
    return cv2.VideoCapture(video_source)

def load_mapping(csv_file):
    mapping, reverse_mapping = {}, {}
    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    digit_seq, char = row[0].strip(), row[1].strip()
                    mapping[digit_seq] = char
                    reverse_mapping[char] = list(digit_seq)
    except FileNotFoundError:
        pass
    return mapping, reverse_mapping

def run_detection(video_source_str='0', model_path='yolo11s-pose.pt', flag_model_path='flag.pt', mapping_csv_path='mapping.csv', current_mode='practice', current_system='chinese', target_sequence=None, start_exam_signal=False, stop_exam_signal=False, is_flag_required=True, session_state=None):
    # -----------------------
    # 初始化設備與效能設定
    # -----------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_gpu = (device == 'cuda')
    print(f"[INFO] 偵測引擎啟動中... 運行設備: {device.upper()}")

    try:
        config_path = get_resource_path('config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] 無法讀取 config.json: {e}")
        return

    ANGLE_TOL_STRICT = config['ANGLE_TOL_STRICT']
    ANGLE_TOL_NORMAL = config['ANGLE_TOL_NORMAL']
    ANGLE_TOL_CANCEL = config['ANGLE_TOL_CANCEL']
    STRAIGHT_ARM_THRESHOLD = config.get('STRAIGHT_ARM_THRESHOLD', 160)
    STRAIGHT_ARM_RATIO_THRESHOLD = config.get('STRAIGHT_ARM_RATIO_THRESHOLD', 0.8)
    MIN_ANGLE_FOR_RATIO_CHECK = config.get('MIN_ANGLE_FOR_RATIO_CHECK', 90)
    STABLE_DELAY = config['STABLE_DELAY']
    READY_ANGLE = config['READY_ANGLE']
    STRAIGHTEN_GRACE_PERIOD = config['STRAIGHTEN_GRACE_PERIOD']
    SMOOTHING_WINDOW_SIZE = config['SMOOTHING_WINDOW_SIZE']
    GESTURE_TIMEOUT = config['GESTURE_TIMEOUT']
    GESTURE_WRIST_DISTANCE_THRESHOLD = config['GESTURE_WRIST_DISTANCE_THRESHOLD']
    GESTURE_CROSS_BUFFER = config['GESTURE_CROSS_BUFFER']
    GESTURE_CROSS_COUNT_THRESHOLD = config['GESTURE_CROSS_COUNT_THRESHOLD']
    TARGET_LOST_TIMEOUT = config['TARGET_LOST_TIMEOUT']
    PERSON_CONF_THRESHOLD = config.get('PERSON_CONF_THRESHOLD', 0.4)
    DISPLAY_WIDTH = config['DISPLAY_WIDTH']
    GRIP_CORNER_DISTANCE_THRESHOLD = 150 

    number_angles = {
        '0': ((180, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)), '1': ((READY_ANGLE, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)),
        '2': ((READY_ANGLE, ANGLE_TOL_STRICT), (90, ANGLE_TOL_STRICT)), '3': ((READY_ANGLE, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)),
        '4': ((READY_ANGLE, ANGLE_TOL_STRICT), (180, ANGLE_TOL_STRICT)), '5': ((135, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)),
        '6': ((90, ANGLE_TOL_STRICT), (READY_ANGLE, ANGLE_TOL_STRICT)), '7': ((45, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)),
        '8': ((315, ANGLE_TOL_NORMAL), (90, ANGLE_TOL_STRICT)), '9': ((300, ANGLE_TOL_NORMAL), (135, ANGLE_TOL_NORMAL))
    }
    navy_angles = {
        'A': ((READY_ANGLE, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)), 'B': ((READY_ANGLE, ANGLE_TOL_STRICT), (90, ANGLE_TOL_STRICT)),
        'C': ((READY_ANGLE, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)), 'D': ((READY_ANGLE, ANGLE_TOL_STRICT), (180, ANGLE_TOL_STRICT)),
        'E': ((135, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)), 'F': ((90, ANGLE_TOL_STRICT), (READY_ANGLE, ANGLE_TOL_STRICT)),
        'G': ((45, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)), 'H': ((315, ANGLE_TOL_NORMAL), (90, ANGLE_TOL_STRICT)),
        'I': ((315, ANGLE_TOL_NORMAL), (135, ANGLE_TOL_NORMAL)), 'J': ((90, ANGLE_TOL_STRICT), (180, ANGLE_TOL_STRICT)),
        'K': ((180, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)), 'L': ((135, ANGLE_TOL_NORMAL), (45, ANGLE_TOL_NORMAL)),
        'M': ((90, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)), 'N': ((45, ANGLE_TOL_NORMAL), (45, ANGLE_TOL_NORMAL)),
        'O': ((270, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)), 'P': ((180, ANGLE_TOL_STRICT), (90, ANGLE_TOL_STRICT)),
        'Q': ((135, ANGLE_TOL_NORMAL), (90, ANGLE_TOL_STRICT)), 'R': ((90, ANGLE_TOL_STRICT), (90, ANGLE_TOL_STRICT)),
        'S': ((45, ANGLE_TOL_NORMAL), (90, ANGLE_TOL_STRICT)), 'T': ((180, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)),
        'U': ((135, ANGLE_TOL_NORMAL), (135, ANGLE_TOL_NORMAL)), 'V': ((45, ANGLE_TOL_NORMAL), (180, ANGLE_TOL_STRICT)),
        'W': ((135, ANGLE_TOL_NORMAL), (270, ANGLE_TOL_STRICT)), 'X': ((135, ANGLE_TOL_NORMAL), (315, ANGLE_TOL_NORMAL)),
        'Y': ((90, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)), 'Z': ((90, ANGLE_TOL_STRICT), (315, ANGLE_TOL_NORMAL)),
        '1': ((READY_ANGLE, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)), '2': ((READY_ANGLE, ANGLE_TOL_STRICT), (90, ANGLE_TOL_STRICT)),
        '3': ((READY_ANGLE, ANGLE_TOL_STRICT), (135, ANGLE_TOL_NORMAL)), '4': ((READY_ANGLE, ANGLE_TOL_STRICT), (180, ANGLE_TOL_STRICT)),
        '5': ((135, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)), '6': ((90, ANGLE_TOL_STRICT), (READY_ANGLE, ANGLE_TOL_STRICT)),
        '7': ((45, ANGLE_TOL_NORMAL), (READY_ANGLE, ANGLE_TOL_STRICT)), '8': ((315, ANGLE_TOL_NORMAL), (90, ANGLE_TOL_STRICT)),
        '9': ((315, ANGLE_TOL_NORMAL), (135, ANGLE_TOL_NORMAL)), '0': ((180, ANGLE_TOL_STRICT), (45, ANGLE_TOL_NORMAL)),
        '#': ((135, ANGLE_TOL_NORMAL), (180, ANGLE_TOL_STRICT))
    }

    def angle_diff(a1, a2): return min(abs(a1-a2), 360-abs(a1-a2))
    def calc_dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) if all(c > 1 for p in [p1,p2] for c in p) else 0.0
    def is_straight_by_ratio(p_s, p_e, p_w, elbow_angle, ratio_thresh, straight_angle_thresh, min_angle):
        # 1. 角度判定：只要手肘角度大於 160°，直接視為伸直
        if elbow_angle >= straight_angle_thresh: return True
        
        # 2. 比例判定：若角度不到 160°，但超過 90° 安全鎖且長度比例夠高，也視為伸直
        if elbow_angle < min_angle: return False
        dist_se, dist_ew, dist_sw = calc_dist(p_s, p_e), calc_dist(p_e, p_w), calc_dist(p_s, p_w)
        return bool((dist_sw / (dist_se + dist_ew)) >= ratio_thresh) if (dist_se + dist_ew) > 0 else False
    def calc_angle_360(p1,p2,p3,hand='left'):
        a_x,a_y,b_x,b_y = p1[0]-p2[0], p1[1]-p2[1], p3[0]-p2[0], p3[1]-p2[1]
        angle = math.degrees(math.atan2(a_x*b_y - a_y*b_x, a_x*b_x + a_y*b_y))
        return float((360 - angle if hand == 'right' else angle + 360) % 360)
    def calc_angle_180(p1,p2,p3):
        if any(c < 1 for p in [p1,p2,p3] for c in p): return 0.0
        v1,v2 = (p1[0]-p2[0],p1[1]-p2[1]), (p3[0]-p2[0],p3[1]-p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1,mag2 = math.sqrt(v1[0]**2+v1[1]**2), math.sqrt(v2[0]**2+v2[1]**2)
        return float(math.degrees(math.acos(max(-1.0,min(1.0, dot/(mag1*mag2)))))) if mag1*mag2 > 0 else 180.0

    def get_correction_info(current, target, tolerance, hand='left'):
        abs_diff = angle_diff(current, target)
        is_ok = bool(abs_diff <= tolerance)
        if is_ok: return True, float(abs_diff), "OK"
        target_is_opposite = (target > 180)
        current_is_opposite = (current > 180)
        if target_is_opposite and not current_is_opposite: advice = "往對面擺"
        elif not target_is_opposite and current_is_opposite: advice = "擺回原側"
        else:
            if not target_is_opposite:
                advice = f"抬{abs_diff:.0f}°" if current < target else f"降{abs_diff:.0f}°"
            else:
                advice = f"降{abs_diff:.0f}°" if current < target else f"抬{abs_diff:.0f}°"
        return False, float(abs_diff), advice

    def recognize_pose(l_angle, r_angle, expected_char=None):
        if angle_diff(l_angle, 45)<=ANGLE_TOL_CANCEL and angle_diff(r_angle,135)<=ANGLE_TOL_CANCEL: return "cancel"
        angles_to_check = navy_angles if current_system == 'navy' else number_angles
        nav_mode = session_state.get('navy_sub_mode', 'ALPHA')
        
        best_match = None
        for sig, ((lt,ll),(rt,rl)) in angles_to_check.items():
            if angle_diff(l_angle,lt)<=ll and angle_diff(r_angle,rt)<=rl:
                if current_system == 'navy' and nav_mode == 'NUMERIC':
                    if sig in ['J', '#']: return sig # 保留切換信號
                    if 'A' <= sig <= 'I':
                        num_map = {'A':'1','B':'2','C':'3','D':'4','E':'5','F':'6','G':'7','H':'8','I':'9'}
                        sig = num_map[sig]
                    elif sig == 'K': sig = '0'
                    elif not sig.isdigit(): continue # 數字模式下，過濾非數字字母
                if current_system == 'navy' and expected_char is not None:
                    if str(sig).upper() == str(expected_char).upper(): return sig
                if best_match is None: best_match = sig
        return best_match

    def is_ready_pose(l,r): return bool(angle_diff(l,READY_ANGLE)<=ANGLE_TOL_STRICT and angle_diff(r,READY_ANGLE)<=ANGLE_TOL_STRICT)
    def is_hands_above_head(kpts):
        if kpts is None or len(kpts)<11: return False
        nose_y, l_w_y, r_w_y = kpts[0][1], kpts[9][1], kpts[10][1]
        # 加嚴判斷：手腕必須明顯高於鼻子 (y 軸數值越小越高)
        return bool(all(y > 0 for y in [nose_y, l_w_y, r_w_y]) and l_w_y < (nose_y - 20) and r_w_y < (nose_y - 20))

    # 載入權重並移至設備 (開啟 FP16 加速)
    try:
        final_model_path = get_resource_path(model_path)
        final_flag_path = get_resource_path(flag_model_path)
        pose_model = YOLO(final_model_path).to(device)
        flag_model = YOLO(final_flag_path).to(device)
    except Exception as e:
        print(f"[ERROR] 無法載入模型: {e}")
        return
    
    # 建立映射表
    mapping, reverse_mapping = {}, {}
    if current_system == 'navy':
        vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        mapping = {char: char for char in vocab}; reverse_mapping = {char: list(char) for char in vocab}
    else:
        final_csv_path = get_resource_path(mapping_csv_path)
        mapping, reverse_mapping = load_mapping(final_csv_path)

    is_camera_off = (str(video_source_str) == "-1")
    if not is_camera_off:
        print(f"[INFO] 嘗試開啟攝影機 ID: {video_source_str}")
        cap = create_video_capture(video_source_str)
        if not cap.isOpened(): 
            print(f"[ERROR] 無法開啟攝影機 ID: {video_source_str}，將強制切換為關閉模式。")
            is_camera_off = True
            cap = None
            frame_width, frame_height = 1280, 720
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] 攝影機開啟成功 (解析度: {frame_width}x{frame_height})")
    else:
        cap = None
        frame_width, frame_height = 1280, 720

    DISPLAY_HEIGHT = int(DISPLAY_WIDTH / (frame_width / frame_height))

    # 狀態變數
    if session_state is None: session_state = {}
    state, state_timer, current_digit, sequence, display_result = "IDLE", 0, None, [], None
    word_history, completed_sequences_stack = [], []
    is_in_challenge_mode, challenge_type = False, "standard"
    challenge_target_string, challenge_current_word_index, current_char_target_sequence, current_char_next_digit_index, challenge_user_sequence, is_error_locked, challenge_invalid_char = "", 0, [], 0, [], False, None
    target_person_id, last_known_target_person_bbox, target_lost_start_time = None, None, 0.0
    cross_sub_state, cross_count, last_gesture_time, gesture_complete = "UNCROSSED", 0, 0, False
    history = {k: deque(maxlen=SMOOTHING_WINDOW_SIZE) for k in ['left_angle','right_angle','left_elbow','right_elbow']}
    arm_straight_timer = 0.0
    frame_counter = 0

    fps_start_time = time.time()
    fps_frame_count = 0
    current_backend_fps = 0
    
    experiment_frame_count = 0
    experiment_accumulated_time = 0.0

    try:
        while True:
            t_start = time.time()
            
            if is_camera_off:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                ret = True
                time.sleep(0.1) # 降低 CPU 使用率
            else:
                ret, frame = cap.read()

            if not ret: break
            current_time = time.time()
            frame_counter += 1
            
            fps_frame_count += 1
            if current_time - fps_start_time >= 1.0:
                current_backend_fps = round(fps_frame_count / (current_time - fps_start_time))
                fps_frame_count = 0
                fps_start_time = current_time
            
            # --- 指令同步 ---
            if session_state.get("new_challenge_string") is not None:
                new_str = session_state["new_challenge_string"]
                c_payload = session_state.get("challenge_payload", {})
                challenge_type = c_payload.get("type", "standard")
                is_in_challenge_mode, word_history, sequence, challenge_user_sequence, is_error_locked = True, [], [], [], False
                session_state["navy_sub_mode"] = "ALPHA" # 重置模式為字母
                
                target_person_id = None
                target_lost_start_time = 0.0
                
                # 初始化考試統計
                if challenge_type == "exam":
                    session_state["exam_stats"] = {"total_signals": 0, "correct_signals": 0}

                invalid_char = next((c for c in new_str if c not in reverse_mapping and not c.isdigit() and c != ','), None)
                if not new_str: state, challenge_target_string, current_char_target_sequence = "CHALLENGE_AWAITING_INPUT", "", []
                elif invalid_char: state, challenge_target_string, challenge_invalid_char = "CHALLENGE_INVALID_CHAR", new_str, invalid_char
                else:
                    state, challenge_target_string, challenge_current_word_index = "IDLE", new_str, 0
                    first_char = new_str[0]
                    if current_system == 'navy' and first_char.isdigit():
                        current_char_target_sequence = ['#']
                    else:
                        current_char_target_sequence = list(reverse_mapping.get(first_char, []))
                    current_char_next_digit_index = 0
                    gesture_complete, cross_count, cross_sub_state = False, 0, "UNCROSSED"
                session_state["new_challenge_string"] = None
            if session_state.get("stop_challenge_mode"):
                is_in_challenge_mode, session_state["stop_challenge_mode"] = False, False
                challenge_target_string, word_history, sequence, challenge_user_sequence, is_error_locked, state = "", [], [], [], False, "IDLE"

            # --- 影像辨識 ---
            pose_results = pose_model.track(frame, persist=True, verbose=False, conf=PERSON_CONF_THRESHOLD, device=device, half=is_gpu)
            all_person_boxes, all_person_kpts = {}, {}
            if pose_results and pose_results[0].boxes is not None and pose_results[0].boxes.id is not None:
                for box, kpts_obj in zip(pose_results[0].boxes, pose_results[0].keypoints):
                    if box.id is not None:
                        p_id = int(box.id.item())
                        all_person_boxes[p_id] = box.xyxy[0].cpu().numpy()
                        all_person_kpts[p_id] = kpts_obj.xy[0].cpu().numpy()
            
            # 目標鎖定邏輯
            if target_person_id is None:
                if is_flag_required:
                    if frame_counter % 5 == 0:
                        flag_res = flag_model.predict(frame, conf=0.5, verbose=False, device=device, half=is_gpu)
                        flag_boxes = flag_res[0].boxes.xyxy.cpu().numpy().tolist() if flag_res and flag_res[0].boxes is not None else []
                        if flag_boxes and all_person_kpts:
                            for p_id, kpts in all_person_kpts.items():
                                l_w, r_w = kpts[9], kpts[10]
                                for fb in flag_boxes:
                                    corners = [(fb[0],fb[1]),(fb[2],fb[1]),(fb[0],fb[3]),(fb[2],fb[3])]
                                    if any(math.sqrt((w[0]-c[0])**2 + (w[1]-c[1])**2) < GRIP_CORNER_DISTANCE_THRESHOLD for w in [l_w,r_w] for c in corners if w[0]>0):
                                        target_person_id = p_id; break
                                if target_person_id: break
                        elif len(all_person_kpts) == 1: target_person_id = list(all_person_kpts.keys())[0]
                else:
                    if all_person_boxes: target_person_id = max(all_person_boxes, key=lambda i: (all_person_boxes[i][2]-all_person_boxes[i][0])*(all_person_boxes[i][3]-all_person_boxes[i][1]))

            # 目標丟失處理
            target_kpts = all_person_kpts.get(target_person_id) if target_person_id else None
            if target_person_id and target_kpts is None:
                if target_lost_start_time == 0.0: target_lost_start_time = current_time
                if (current_time - target_lost_start_time) > TARGET_LOST_TIMEOUT:
                    target_person_id, state = None, "IDLE"
                    target_lost_start_time = 0.0 # 關鍵修復：觸發超時後必須歸零，避免下次一閃爍就秒殺
                    # 若在測驗中遺失目標，強制終止測驗，防止換人數據累加
                    if is_in_challenge_mode:
                        is_in_challenge_mode = False
                        challenge_target_string, current_char_target_sequence, challenge_user_sequence, word_history = "", [], [], []
                        is_error_locked = False
                        if session_state and "exam_stats" in session_state:
                            del session_state["exam_stats"]
            elif target_kpts is not None:
                target_lost_start_time = 0.0
                try:
                    p_idx = list(pose_results[0].boxes.id.cpu().numpy()).index(target_person_id)
                    last_known_target_person_bbox = pose_results[0].boxes[p_idx].xyxy[0].cpu().numpy().tolist()
                except: pass
                for k, f in {'left_angle':lambda:calc_angle_360(target_kpts[9],target_kpts[5],target_kpts[11],'left'), 'right_angle':lambda:calc_angle_360(target_kpts[10],target_kpts[6],target_kpts[12],'right'), 'left_elbow':lambda:calc_angle_180(target_kpts[5],target_kpts[7],target_kpts[9]), 'right_elbow':lambda:calc_angle_180(target_kpts[6],target_kpts[8],target_kpts[10])}.items():
                    history[k].append(f())

            # --- 核心邏輯運算 ---
            angs, lok_arm, rok_arm, h_up, arms_stable_bent = {}, False, False, False, False
            if target_kpts is not None and len(history['left_angle']) >= SMOOTHING_WINDOW_SIZE:
                angs = {k: float(np.mean(v)) for k, v in history.items()}
                lok_arm = bool(is_straight_by_ratio(target_kpts[5], target_kpts[7], target_kpts[9], angs['left_elbow'], STRAIGHT_ARM_RATIO_THRESHOLD, STRAIGHT_ARM_THRESHOLD, MIN_ANGLE_FOR_RATIO_CHECK))
                rok_arm = bool(is_straight_by_ratio(target_kpts[6], target_kpts[8], target_kpts[10], angs['right_elbow'], STRAIGHT_ARM_RATIO_THRESHOLD, STRAIGHT_ARM_THRESHOLD, MIN_ANGLE_FOR_RATIO_CHECK))
                h_up = bool(is_hands_above_head(target_kpts))
                if not (lok_arm and rok_arm):
                    if arm_straight_timer == 0.0: arm_straight_timer = current_time
                else: arm_straight_timer = 0.0
                arms_stable_bent = bool(arm_straight_timer > 0 and (current_time - arm_straight_timer) > 1.5)

                if state in ["IDLE", "CHALLENGE_READY_TO_END", "READY", "COOLDOWN", "DETECTING", "GRACE_PERIOD"]:
                    if gesture_complete and not h_up:
                        if state == "CHALLENGE_READY_TO_END": 
                            state, state_timer = "CHALLENGE_COMPLETE_PROMPT", time.time()
                        elif not is_in_challenge_mode and len(word_history) > 0 and state != "IDLE":
                            state, state_timer = "CHALLENGE_COMPLETE_PROMPT", time.time()
                        elif state == "IDLE": 
                            state = "WAITING"
                            word_history.clear()
                        else: 
                            state = "IDLE"
                            word_history.clear()
                            
                        gesture_complete, cross_count, cross_sub_state = False, 0, "UNCROSSED"
                        sequence.clear(); challenge_user_sequence.clear(); current_digit, display_result, is_error_locked = None, None, False
                        if is_in_challenge_mode and state == "WAITING":
                            challenge_current_word_index = 0
                            first_char = challenge_target_string[0] if len(challenge_target_string) > 0 else ""
                            if current_system == 'navy' and first_char.isdigit():
                                current_char_target_sequence = ['#']
                            else:
                                current_char_target_sequence = list(reverse_mapping.get(first_char, []))
                            current_char_next_digit_index = 0
                            if challenge_type == "exam" and "exam_stats" in session_state:
                                session_state["exam_stats"]["correct_signals"] = 0
                                session_state["exam_stats"]["has_errored_on_current_signal"] = False
                                session_state["exam_stats"]["total_signals"] = 0
                            if challenge_type == "exam" and "exam_stats" in session_state:
                                session_state["exam_stats"]["correct_signals"] = 0
                                session_state["exam_stats"]["has_errored_on_current_signal"] = False
                    elif h_up and not gesture_complete:
                        if time.time() - last_gesture_time > GESTURE_TIMEOUT: cross_count, cross_sub_state = 0, "UNCROSSED"
                        is_c = abs(target_kpts[9][0] - target_kpts[10][0]) < GESTURE_WRIST_DISTANCE_THRESHOLD
                        is_uc = abs(target_kpts[9][0] - target_kpts[10][0]) > (GESTURE_WRIST_DISTANCE_THRESHOLD + GESTURE_CROSS_BUFFER)
                        if cross_sub_state == "UNCROSSED" and is_c: cross_count += 1; cross_sub_state = "CROSSED"; last_gesture_time = time.time()
                        if cross_count >= GESTURE_CROSS_COUNT_THRESHOLD: gesture_complete = True
                        elif cross_sub_state == "CROSSED" and is_uc: cross_sub_state = "UNCROSSED"; last_gesture_time = time.time()
                elif not h_up and not is_error_locked and state not in ["IDLE", "CHALLENGE_READY_TO_END", "WAITING", "READY", "DETECTING", "GRACE_PERIOD", "COOLDOWN", "CHALLENGE_AWAITING_INPUT", "CHALLENGE_INVALID_CHAR", "CHALLENGE_AWAITING_GESTURE", "CHALLENGE_COMPLETE_PROMPT"]:
                    state = "IDLE"

                # 關鍵修正：呼叫 recognize_pose 時傳入期望值
                expected = None
                if is_in_challenge_mode and current_char_target_sequence:
                    expected = current_char_target_sequence[current_char_next_digit_index]
                
                det_p = recognize_pose(angs['left_angle'], angs['right_angle'], expected_char=expected) if state in ["READY", "DETECTING", "GRACE_PERIOD"] else None
                
                if state == "WAITING" and is_ready_pose(angs['left_angle'], angs['right_angle']): state = "READY"
                elif state == "READY" and det_p is not None: state, current_digit, state_timer = "DETECTING", det_p, time.time()
                elif state == "DETECTING":
                    if det_p != current_digit: state, current_digit = "READY", None
                    elif time.time() - state_timer >= STABLE_DELAY: state, state_timer = "GRACE_PERIOD", time.time()
                elif state == "GRACE_PERIOD":
                    if det_p != current_digit: state, current_digit = "READY", None
                    else:
                        is_v = (current_digit == "cancel") or (lok_arm and rok_arm)
                        if is_in_challenge_mode and challenge_type == "teaching":
                            is_v = (current_digit == "cancel") or (str(current_digit) == str(current_char_target_sequence[current_char_next_digit_index]))
                        
                        if is_v:
                            nx_s, is_switch_signal = "COOLDOWN", False
                            if current_system == 'navy':
                                if current_digit == '#': session_state["navy_sub_mode"] = "NUMERIC"; is_switch_signal = True
                                elif current_digit == 'J' and session_state.get("navy_sub_mode") == "NUMERIC": session_state["navy_sub_mode"] = "ALPHA"; is_switch_signal = True

                            if is_in_challenge_mode:
                                if current_digit == "cancel":
                                    if is_error_locked and challenge_user_sequence: challenge_user_sequence.pop()
                                    is_error_locked, current_char_next_digit_index = False, len(challenge_user_sequence)
                                elif is_switch_signal:
                                    challenge_user_sequence, current_char_next_digit_index = [], 0
                                elif not is_error_locked or challenge_type == "teaching":
                                    if challenge_type == "teaching":
                                        challenge_user_sequence.append(str(current_digit)); current_char_next_digit_index += 1
                                    else:
                                        challenge_user_sequence.append(str(current_digit)); cl = len(challenge_user_sequence)
                                        
                                        if challenge_type == "exam" and "exam_stats" in session_state:
                                            session_state["exam_stats"]["total_signals"] += 1
                                            
                                        if "".join(challenge_user_sequence) != "".join(current_char_target_sequence[:cl]): 
                                            is_error_locked = True
                                        else: 
                                            current_char_next_digit_index = cl
                                            if challenge_type == "exam" and "exam_stats" in session_state:
                                                session_state["exam_stats"]["correct_signals"] += 1
                                
                                # 判斷是否完成目前步驟
                                if (current_char_next_digit_index == len(current_char_target_sequence)) or is_switch_signal:
                                    if not is_switch_signal and (current_char_next_digit_index == len(current_char_target_sequence)):
                                        word_history.append(challenge_target_string[challenge_current_word_index]); challenge_current_word_index += 1
                                        
                                        # 跳過分隔符號
                                        while challenge_current_word_index < len(challenge_target_string) and challenge_target_string[challenge_current_word_index] == ',':
                                            word_history.append(',')
                                            challenge_current_word_index += 1
                                            
                                    challenge_user_sequence, current_char_next_digit_index = [], 0
                                    if challenge_current_word_index >= len(challenge_target_string):
                                        nx_s = "CHALLENGE_AWAITING_GESTURE"; current_char_target_sequence = []
                                    else:
                                        next_char = challenge_target_string[challenge_current_word_index]
                                        curr_mode = session_state.get("navy_sub_mode", "ALPHA")
                                        if current_system == 'navy':
                                            if next_char.isdigit() and curr_mode == "ALPHA": current_char_target_sequence = ['#']
                                            elif not next_char.isdigit() and curr_mode == "NUMERIC": current_char_target_sequence = ['J']
                                            else: current_char_target_sequence = list(reverse_mapping.get(next_char, []))
                                        else: current_char_target_sequence = list(reverse_mapping.get(next_char, []))
                            else:
                                if not is_switch_signal:
                                    tsl = 1 if current_system == 'navy' else 4
                                    if current_digit == "cancel":
                                        if sequence: sequence.pop()
                                        elif word_history and completed_sequences_stack:
                                            word_history.pop(); last_seq = completed_sequences_stack.pop(); sequence.clear(); sequence.extend(last_seq[:-1])
                                    elif len(sequence) < tsl: sequence.append(str(current_digit))
                            state, state_timer = nx_s, time.time()
                        elif time.time() - state_timer > STRAIGHTEN_GRACE_PERIOD: state, current_digit = "READY", None
                        elif time.time() - state_timer > STRAIGHTEN_GRACE_PERIOD: state, current_digit = "READY", None
                elif state in ["COOLDOWN", "CHALLENGE_AWAITING_GESTURE"] and is_ready_pose(angs['left_angle'], angs['right_angle']): state, current_digit = ("READY" if state == "COOLDOWN" else "CHALLENGE_READY_TO_END"), None
                elif state == "CHALLENGE_COMPLETE_PROMPT" and time.time() - state_timer > 2.0:
                    word_history.clear()
                    if not is_in_challenge_mode:
                        state = "IDLE"
                    elif challenge_type == "exam":
                        state = "IDLE"
                        is_in_challenge_mode = False
                        if "exam_stats" in session_state: del session_state["exam_stats"]
                    else:
                        state = "CHALLENGE_AWAITING_INPUT"

                # --- 畫面提示文字 (Prompt) ---
            corr_info = {"target_signal":None,"target_l_angle":None,"target_r_angle":None,"l_angle_diff":None,"r_angle_diff":None,"l_angle_ok":False,"r_angle_ok":False,"l_arm_straight_ok":False,"r_arm_straight_ok":False,"l_advice":"-","r_advice":"-","is_correct":False}
            if angs and is_in_challenge_mode and challenge_type == "teaching" and current_char_target_sequence and state in ["READY", "DETECTING", "GRACE_PERIOD"] and (angs['left_angle'] > 30 or angs['right_angle'] > 30):
                eff_t = current_char_target_sequence[current_char_next_digit_index]
                ta = (navy_angles if current_system == 'navy' else number_angles).get(eff_t)
                if ta:
                    lok, ld, la = get_correction_info(angs['left_angle'], ta[0][0], ta[0][1], hand='left')
                    rok, rd, ra = get_correction_info(angs['right_angle'], ta[1][0], ta[1][1], hand='right')
                    corr_info.update({"target_signal":eff_t,"target_l_angle":float(ta[0][0]),"target_r_angle":float(ta[1][0]),"l_angle_diff":float(ld),"r_angle_diff":float(rd),"l_angle_ok":bool(lok),"r_angle_ok":bool(rok),"l_arm_straight_ok":lok_arm,"r_arm_straight_ok":rok_arm,"l_advice":la,"r_advice":ra,"is_correct":bool(lok and rok and lok_arm and rok_arm)})

            pc = None
            if not target_person_id: pc = "尋找目標 (旗手)..."
            elif target_kpts is None: pc = "目標遺失，請回畫面"
            elif state == "CHALLENGE_INVALID_CHAR": pc = f"字元 '{challenge_invalid_char}' 不明"
            elif state == "CHALLENGE_AWAITING_INPUT": pc = "請設定練習字串"
            elif state == "CHALLENGE_COMPLETE_PROMPT": pc = "練習完成！"
            elif state == "CHALLENGE_AWAITING_GESTURE": pc = "完成！請雙手放下"
            elif state == "CHALLENGE_READY_TO_END": pc = "請做出結束手勢"
            elif is_in_challenge_mode and challenge_type == "teaching" and state in ["READY", "DETECTING", "GRACE_PERIOD"] and (angs.get('left_angle',0)>30 or angs.get('right_angle',0)>30):
                if arms_stable_bent: pc = "請將手臂伸直"
                elif corr_info["is_correct"]: pc = "姿勢正確！請維持..."
                else:
                    la, ra = corr_info["l_advice"], corr_info["r_advice"]
                    pc = f"左{la}, 右{ra}" if la!="OK" and ra!="OK" else (f"左{la}" if la!="OK" else f"右{ra}")
            elif is_error_locked: pc = "輸入錯誤！請雙手放下" if not is_ready_pose(angs.get('left_angle',0), angs.get('right_angle',0)) else "請比出 [取消] 手勢"
            elif state == "IDLE":
                if is_in_challenge_mode:
                    pc = "請舉起雙手交叉以開始測驗"
                else:
                    pc = "請從左側選擇模式，或交叉雙手自由練習"
            elif state == "WAITING": pc = "雙手放下預備"
            elif state == "READY":
                if is_in_challenge_mode and current_char_target_sequence: 
                    if challenge_type == "exam":
                        pc = "請比出下一個動作"
                    else:
                        pc = f"請比出 {current_char_target_sequence[current_char_next_digit_index]}"
                else: pc = "準備就緒，開始比劃"
            elif state == "DETECTING": pc = f"偵測到 {current_digit}..." if current_digit else "偵測中..."
            elif state == "GRACE_PERIOD": pc = "判定中..."
            elif state == "COOLDOWN": pc = "成功！請放下雙手"

            detection_data = {
                "state": state, "prompt_code": pc, "cross_count": int(cross_count),
                "left_angle": angs.get('left_angle'), "right_angle": angs.get('right_angle'),
                "current_digit": current_digit, "l_arm_status": "伸直" if lok_arm else "彎曲", "r_arm_status": "伸直" if rok_arm else "彎曲",
                "sequence": list(challenge_user_sequence if is_in_challenge_mode else sequence),
                "display_result": display_result, "target_person_bbox": last_known_target_person_bbox if target_person_id else None,
                "flag_boxes": [], "mode": current_mode, "word_history": list(word_history),
                "challenge_info": {"is_challenge_mode":bool(is_in_challenge_mode),"challenge_type":challenge_type,"target_string":challenge_target_string,"current_word_index":int(challenge_current_word_index),"current_char_target_sequence":current_char_target_sequence,"current_char_next_digit_index":int(current_char_next_digit_index),"is_error_locked":bool(is_error_locked)},
                "correction_data": corr_info,
                "exam_stats": session_state.get("exam_stats"),
                "compute_device": "CUDA" if is_gpu else "CPU",
                "backend_fps": current_backend_fps
            }

            if current_mode == 'practice' and not is_in_challenge_mode and len(sequence) == (1 if current_system == 'navy' else 4):
                res = mapping.get("".join(sequence), "?")
                word_history.append(res); completed_sequences_stack.append(list(sequence)); sequence.clear()
                detection_data["display_result"] = res

            rs_f = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            if target_person_id and last_known_target_person_bbox:
                b = last_known_target_person_bbox
                cv2.rectangle(rs_f, (int(b[0]*DISPLAY_WIDTH/frame_width), int(b[1]*DISPLAY_HEIGHT/frame_height)), (int(b[2]*DISPLAY_WIDTH/frame_width), int(b[3]*DISPLAY_HEIGHT/frame_height)), (255,0,0), 2)
            _, jpeg = cv2.imencode('.jpg', rs_f)

            t_end = time.time()
            experiment_accumulated_time += (t_end - t_start)
            experiment_frame_count += 1
            if experiment_frame_count >= 300:
                avg_latency_ms = (experiment_accumulated_time / 300) * 1000
                avg_fps = 300.0 / experiment_accumulated_time if experiment_accumulated_time > 0 else 0
                try:
                    with open("performance_log.txt", "a", encoding="utf-8") as lf:
                        lf.write(f"[效能測試] 累積 300 幀 | 設備: {'CUDA' if is_gpu else 'CPU'} | 平均推論延遲: {avg_latency_ms:.2f} ms | 平均 FPS: {avg_fps:.2f}\n")
                except Exception:
                    pass
                experiment_frame_count = 0
                experiment_accumulated_time = 0.0

            yield jpeg.tobytes(), detection_data
    finally:
        if cap: cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='0'); parser.add_argument('--model_path', type=str, default='yolo11s-pose.pt')
    parser.add_argument('--flag_model_path', type=str, default='flag.pt'); parser.add_argument('--mapping_csv', type=str, default='mapping.csv')
    args = parser.parse_args()
    for f, d in run_detection(video_source_str=args.video_source, model_path=args.model_path, flag_model_path=args.flag_model_path, mapping_csv_path=args.mapping_csv): pass
