import time
import argparse
from yolo_logic import run_detection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='0')
    parser.add_argument('--model_path', type=str, default='yolo11s-pose.pt')
    parser.add_argument('--flag_model_path', type=str, default='flag.pt')
    parser.add_argument('--mapping_csv', type=str, default='mapping.csv')
    args = parser.parse_args()

    print("[效能測試] 啟動外測腳本...")
    gen = run_detection(video_source_str=args.video_source, model_path=args.model_path, flag_model_path=args.flag_model_path, mapping_csv_path=args.mapping_csv)
    
    experiment_frame_count = 0
    experiment_accumulated_time = 0.0

    try:
        while True:
            t_start = time.time()
            next(gen)
            t_end = time.time()
            
            experiment_accumulated_time += (t_end - t_start)
            experiment_frame_count += 1
            
            if experiment_frame_count == 300:
                avg_latency_ms = (experiment_accumulated_time / 300) * 1000
                avg_fps = 1.0 / (experiment_accumulated_time / 300) if experiment_accumulated_time > 0 else 0
                print(f"==================================================")
                print(f"[效能測試] 累積 300 幀 | 平均推論延遲: {avg_latency_ms:.2f} ms | 平均 FPS: {avg_fps:.2f}")
                print(f"==================================================")
                experiment_frame_count = 0
                experiment_accumulated_time = 0.0
    except StopIteration:
        print("[效能測試] 結束。")
    except KeyboardInterrupt:
        print("\n[效能測試] 使用者中斷。")

if __name__ == '__main__':
    main()
