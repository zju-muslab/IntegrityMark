def evaluate_tampering_overlap_ratio(detection_segs, tamper_segs):
    """
    评估篡改检测性能，基于区间重叠比例计算precision和recall
    
    Args:
        detection_segs: 检测到的段落列表，每个元素为[start, end, label]
        tamper_segs: 实际篡改的段落列表，每个元素为[start, end, label]
    
    Returns:
        precision: 检测结果中正确识别的比例
        recall: 实际篡改段落中被正确识别的比例
    """
    # 提取所有篡改段落(标签为1)
    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]
    
    # 只考虑≥100ms的检测为positive (8000样本点约等于100ms)
    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1 and seg[1] - seg[0] >= 8000]
    
    # 如果没有检测结果或实际篡改，直接返回
    if not detection_segs_1 and not tamper_segs_1:
        return 1.0, 1.0, 1.0, 0.0  # 都是空的情况下，认为完美匹配，TOE为0
    
    if not detection_segs_1:
        return 0.0, 0.0, 0.0, float('inf')  # 没有检测结果但有篡改，TOE为无穷大
    
    if not tamper_segs_1:
        return 0.0, 1.0, 0.0, float('inf')  # 有检测结果但没有篡改，TOE为无穷大
    
    # 计算所有检测段落的总长度
    total_detected_length = sum(seg[1] - seg[0] for seg in detection_segs_1)
    
    # 计算所有实际篡改段落的总长度
    total_tamper_length = sum(seg[1] - seg[0] for seg in tamper_segs_1)
    
    # 计算正确检测的重叠长度
    total_overlap_length = 0
    
    # 对每个检测段落，计算与所有实际篡改段落的重叠
    for detect_seg in detection_segs_1:
        detect_start, detect_end = detect_seg[0], detect_seg[1]
        
        for tamper_seg in tamper_segs_1:
            tamper_start, tamper_end = tamper_seg[0], tamper_seg[1]
            
            # 计算重叠部分
            overlap_start = max(detect_start, tamper_start)
            overlap_end = min(detect_end, tamper_end)
            
            # 如果有重叠
            if overlap_start < overlap_end:
                total_overlap_length += (overlap_end - overlap_start)
    
    # 计算precision和recall
    # precision: 重叠长度占检测总长度的比例
    precision = total_overlap_length / total_detected_length if total_detected_length > 0 else 0
    
    # recall: 重叠长度占实际篡改总长度的比例
    recall = total_overlap_length / total_tamper_length if total_tamper_length > 0 else 0
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算时间偏移误差(TOE)
    toe_values = []
    
    # 对每个检测段落，计算与最近实际篡改段落的边界误差
    for detect_seg in detection_segs_1:
        detect_start, detect_end = detect_seg[0], detect_seg[1]
        
        # 找到最近的左边界
        min_left_error = float('inf')
        for tamper_seg in tamper_segs_1:
            tamper_start = tamper_seg[0]
            error = abs(detect_start - tamper_start)
            if error < min_left_error:
                min_left_error = error
        
        # 找到最近的右边界
        min_right_error = float('inf')
        for tamper_seg in tamper_segs_1:
            tamper_end = tamper_seg[1]
            error = abs(detect_end - tamper_end)
            if error < min_right_error:
                min_right_error = error
        
        # 添加到TOE值列表
        if min_left_error != float('inf'):
            toe_values.append(min_left_error)
        if min_right_error != float('inf'):
            toe_values.append(min_right_error)
    
    # 计算平均TOE
    toe = sum(toe_values) / len(toe_values) if toe_values else float('inf')
    
    return precision, recall, f1, toe






def evaluate_tampering_boundary_accuracy(detection_segs, tamper_segs):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    le = []  # 用于存储边界误差
   
    # 提取所有状态为1的段
    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]
    
    # 只考虑≥100ms的检测为positive
    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1 and seg[1] - seg[0] >= 8000]
    
    # 跟踪已成为TP的检测
    tp_detection_indices = set()
    
    # 计算TP和FN
    for tamper_seg in tamper_segs_1:
        found_match = False
        min_error = float('inf')
        best_index = -1
        best_head_error = float('inf')
        best_tail_error = float('inf')
       
        # 查找最佳匹配
        for i, detect_seg in enumerate(detection_segs_1):
            # 检查是否有重叠
            if detect_seg[0] < tamper_seg[1] and detect_seg[1] > tamper_seg[0]:
                head_error = abs(tamper_seg[0] - detect_seg[0])
                tail_error = abs(tamper_seg[1] - detect_seg[1])
                total_error = head_error + tail_error
               
                if total_error < min_error:
                    min_error = total_error
                    best_index = i
                    best_head_error = head_error
                    best_tail_error = tail_error
                    found_match = True
       
        # 只有当头尾误差均<100ms时才是TP
        if found_match and best_head_error < 1600 and best_tail_error < 1600:
            TP += 1
            tp_detection_indices.add(best_index)
            le.append(best_head_error)
            le.append(best_tail_error)
        else:
            FN += 1
   
    # 所有不是TP的positive检测都是FP
    FP = len(detection_segs_1) - len(tp_detection_indices)
   
    # 计算TN：如果没有检测报告且没有实际篡改
    if not detection_segs_1 and not tamper_segs_1:
        TN = 1
   
    return TP, FN, TN, FP, le

