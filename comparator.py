def compare_operations(video_ops, standard_ops):
    """
    对比视频操作和标准操作
    返回报告列表
    """
    report = []
    step_index = 0

    for op in video_ops:
        if step_index >= len(standard_ops):
            break

        std_op = standard_ops[step_index]

        if op['type'] == "error":
            report.append({
                "step": step_index + 1,
                "expected": std_op,
                "video": op,
                "result": "报错",
                "remark": op['content']
            })
            continue

        if op['type'] == std_op['type'] and (std_op.get("target", "") in op['content']):
            report.append({
                "step": step_index + 1,
                "expected": std_op,
                "video": op,
                "result": "正确"
            })
            step_index += 1
        else:
            report.append({
                "step": step_index + 1,
                "expected": std_op,
                "video": op,
                "result": "错误"
            })
            step_index += 1

    # 剩余未完成步骤
    for i in range(step_index, len(standard_ops)):
        report.append({
            "step": i + 1,
            "expected": standard_ops[i],
            "video": None,
            "result": "未完成"
        })

    return report
