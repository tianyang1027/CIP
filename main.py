import json
from video_processing import extract_video_operations
from document_parser import load_standard_operations
from comparator import compare_operations

VIDEO_PATH = "sample_data/recording.mp4"
STANDARD_OPS_PATH = "sample_data/standard_ops.json"

def main():
    # 1. 提取视频操作事件
    video_ops = extract_video_operations(VIDEO_PATH)
    print("视频操作事件提取完成")
    
    # 2. 加载标准操作
    standard_ops = load_standard_operations(STANDARD_OPS_PATH)
    print("标准操作加载完成")

    # 3. 对比视频操作和标准操作
    report = compare_operations(video_ops, standard_ops)

    # 4. 输出报告
    with open("operation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("对比报告生成完成: operation_report.json")

if __name__ == "__main__ ":
    main()
