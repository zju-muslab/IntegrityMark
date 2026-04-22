from collections import deque
import torch  # 仅当 state 需要 tensor 操作时



def seg_results(pred_segs, discrete_points, debug=False):
    """
    计算篡改检测的性能指标 (TP/TN/FP/FN) 和定位误差 (LE)

    参数:
        pred_segs: 检测器预测的篡改时间段列表，每个元素为 [start, end, state]
        discrete_points: 真实篡改位置列表（样本索引）
        debug: 是否打印调试信息

    返回:
        TP: True Positive - 正确检测到的篡改位置数
        FP: False Positive - 误报的检测位置数
        TN: True Negative - 正确识别的未篡改情况
        FN: False Negative - 漏检的篡改位置数
        LE: Location Error - 定位误差列表

    【LE 单位说明】
        - LE 中的每个值为 "样本数"（samples）
        - 对于 16kHz 采样率的音频：
          * 3200 samples = 0.2 秒 (3200 / 16000)
          * 16000 samples = 1.0 秒
          * 阈值 3200 samples = 200 毫秒（用于判断检测点是否有效）

    【检测逻辑】
        1. 对每个真实篡改位置 (discrete_points)，找最接近的预测位置 (pred_points)
        2. 如果距离 <= 3200 samples（200ms），判定为 TP，记录定位误差
        3. 如果无匹配的预测位置，则为 FN
        4. 无法匹配的预测位置最多计为 3 个 FP
        5. 如果没有任何篡改位置和预测位置，则为 TN
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    le = []  # Location Error - 定位误差列表，单位为样本数 (samples)

    # 提取预测的篡改点位置（起始位置和结束位置）
    pred_points = get_invalid_points_robust(pred_segs, debug=debug)

    if debug:
        print(pred_points, discrete_points)
    # print(pred_segs,pred_points,discrete_points)

    # 对每个真实篡改位置，查找最接近的预测位置
    for p1 in discrete_points:
        # 计算与所有预测位置的距离
        matches = [(abs(p2 - p1), p2) for p2 in pred_points]

        if matches:  # 如果有候选预测点
            min_dist, best_match = min(matches)  # 找到距离最近的预测点

            # 判断检测是否有效：距离必须在阈值 3200 样本内（约 200ms）
            if min_dist <= 3200:
                TP += 1  # 检测成功
                le.append(min_dist)  # 记录定位误差（单位：样本数）
                pred_points.remove(best_match)  # 移除已匹配的预测点
        else:
            # 没有预测点与该真实位置匹配
            FN += 1

    # 剩余未匹配的预测点为误报（FP），最多计 3 个
    FP = min(3, len(pred_points))

    # 如果既没有真实篡改位置，也没有预测位置，则为真阴性
    if len(discrete_points) == 0 and len(pred_points) == 0:
        TN += 1

    return TP, FP, TN, FN, le
            
def get_invalid_points_robust(
    detection_segs,
    tolerated_errors=64,
    history_size=8,
    check_window=5,  # 检查窗口大小
    length_increment=32,
    min_length=512,
    max_length=2064,  # Updated to match paper specification (IntegrityMark paper, Section V.A)
    state_bit_size=4,
    debug=False
):

    def get_state_num(state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().tolist()
        return int(''.join(map(str, state)), 2)
    

    def next_expected_state(current_state):
        """计算下一个预期状态（循环二进制递增）"""
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.cpu().tolist()
        binary_str = ''.join(map(str, current_state))
        current_num = int(binary_str, 2)
        next_num = (current_num + 1) % (2 ** state_bit_size)
        return [int(b) for b in bin(next_num)[2:].zfill(state_bit_size)]

    def next_expected_length(current_length):
        """计算下一个预期长度，确保长度在 min_length 和 max_length 之间循环"""
        next_length = current_length + length_increment
        return next_length if next_length <= max_length else next_length - max_length + min_length

    invalid_points = []
    history_states = deque(maxlen=history_size)  # 维护历史状态
    history_lengths = deque(maxlen=history_size)  # 维护历史长度
    consecutive_errors = 0
    reset = True
    i = 0
    while i < len(detection_segs) - check_window:
        if i<10:
            loop_size=history_size
        else:
            loop_size=history_size+2
        # 初始化历史信息
        if reset:
            for _ in range(loop_size):
                start, end, state = detection_segs[i]
                length=end-start
                if min_length<length<max_length:
                    history_lengths.append(length)
                else:
                    history_lengths.append(-1)
                history_states.append(state)
                reset = False
                i += 1
                if i >= len(detection_segs) - check_window:
                    break
            continue

        # 当前段信息
        start, end, state = detection_segs[i]
        current_length = end - start

        # 处理历史长度异常情况
        if min(history_lengths) < 1000 and max(history_lengths) > 1800:
            for idx_h, length in enumerate(history_lengths):
                if 0<= length < 1000:
                    history_lengths[idx_h] = length - min_length + max_length
                else:
                    history_lengths[idx_h] = length

        expected_lengths=[]
        for idx, length in enumerate(history_lengths):
            if length!=-1:
                state1=get_state_num(history_states[idx])
                state2=get_state_num(history_states[-1])
                if state2>=state1:
                    times=state2-state1+1
                else:
                    times=state2+16-state1+1
                expected_lengths.append(length + length_increment * times)

                
        # expected_next_length = sum(expected_lengths) / len(expected_lengths)
        # 去掉最大值和最小值（确保至少有两个元素进行平均）

        valid_expected_lengths = [length for length in expected_lengths if length != -1]
        if len(valid_expected_lengths)==0:
            reset=True
            i+=1
            continue

        elif len(valid_expected_lengths) > 3:
            trimmed_lengths = sorted(valid_expected_lengths)[1:-1]  # 去掉最大和最小的值
        else:
            trimmed_lengths = valid_expected_lengths  # 如果元素太少，直接使用原始数据

        expected_next_length = sum(trimmed_lengths) / len(trimmed_lengths)

        if expected_next_length > max_length:
            expected_next_length = expected_next_length - max_length + min_length

        # 计算预期状态
        expected_next_state = next_expected_state(history_states[-1])

        # 修正超出最大长度的历史长度值
        history_lengths = deque(
            [length - max_length + min_length if length > max_length else length for length in history_lengths],
            maxlen=history_size
        )

        history_invalid_cnt = 0
        for length in reversed(history_lengths):
            if length == -1:
                history_invalid_cnt += 1
            else:
                break
        invalid_cnt=history_invalid_cnt
        # 预测未来 `check_window` 个状态和长度
        future_states = [expected_next_state]
        future_length=expected_next_length
        future_lengths = [future_length]
        for _ in range(check_window - 1):
            expected_next_state = next_expected_state(expected_next_state)
            future_length = next_expected_length(future_length)
            future_states.append(expected_next_state)

            future_lengths.append(future_length)

        # 进行检查
        validation_results = []
        pred_seg_lengths= []
        pred_seg_states=[]
        if debug:
            print(f'--- Start: {start}, Expected State: {future_states[0]}, Expected Length: {future_lengths[0]}, History Lengths: {trimmed_lengths}({expected_lengths})[{history_lengths}]')
        last_valid_idx=-1
        for j in range(i, min(i + check_window, len(detection_segs))):
            seg_start, seg_end, seg_state = detection_segs[j]
            seg_length = seg_end - seg_start
            pred_seg_lengths.append(seg_length)
            pred_seg_states.append(seg_state)
            # 检查状态是否匹配
            state_valid = any(seg_state.tolist() == future_state for future_state in future_states[max(last_valid_idx+1,j-i-2):j-i+2])

            

            if state_valid:
                matching_index = future_states.index(seg_state.tolist())
                expected_length = future_lengths[matching_index]
                length_valid = abs(seg_length - expected_length) <= tolerated_errors
            else:
                length_valid = False
                invalid_cnt+=1
            
            segment_valid=state_valid and length_valid
            if segment_valid:
                last_valid_idx=j-i

            validation_results.append(segment_valid)

            if debug:
                print(f'({i},{j}),{[seg_start,seg_end]} Valid: {state_valid and length_valid}({state_valid},{length_valid}), State: {seg_state.tolist()}, Length: {seg_length}', end=', ')
                print(f'Expected Length: {expected_length:.1f}' if state_valid else 'Not Valid')

        # 处理无效点

        if invalid_cnt>tolerated_errors:
            invalid_points.append(detection_segs[i-invalid_cnt][0])
            reset = True
            
        elif not any(validation_results):
            invalid_points.append(start)
            reset = True
        else:
            reset = False

        update_times=0
        for v_idx, val in enumerate(validation_results):
            history_states.append(pred_seg_states[v_idx])
            state = detection_segs[i + v_idx][2]
            update_times+=1
            if val:
                history_lengths.append(pred_seg_lengths[v_idx])
                break  
            else:
                history_lengths.append(-1)
        i+=update_times

    # 合并不连续点
    invalid_points = sorted(invalid_points)

    return invalid_points
