from collections import deque
import torch

from .dual_sliding_window import DualSlidingWindowVerifier



def seg_results(
    pred_segs,
    discrete_points,
    debug=False,
    *,
    manifest=None,
    master_key=None,
    history_size=8,
    check_window=5,
    tolerated_errors=64,
):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    le = []


    pred_points = get_invalid_points_robust(
        pred_segs,
        debug=debug,
        manifest=manifest,
        master_key=master_key,
        history_size=history_size,
        check_window=check_window,
        tolerated_errors=tolerated_errors,
    )

    if debug:
        print(pred_points, discrete_points)



    for p1 in discrete_points:

        matches = [(abs(p2 - p1), p2) for p2 in pred_points]

        if matches:
            min_dist, best_match = min(matches)


            if min_dist <= 3200:
                TP += 1
                le.append(min_dist)
                pred_points.remove(best_match)
            else:
                FN += 1
        else:

            FN += 1


    FP = len(pred_points)


    if len(discrete_points) == 0 and len(pred_points) == 0:
        TN += 1

    return TP, FP, TN, FN, le

def get_invalid_points_robust(
    detection_segs,
    tolerated_errors=64,
    history_size=8,
    check_window=5,
    length_increment=32,
    min_length=512,
    max_length=2064,
    state_bit_size=4,
    debug=False,
    manifest=None,
    master_key=None,
):

    if manifest is not None and manifest.schedule_version == "keyed_prf_v1":
        if master_key is None:
            raise ValueError("master_key is required for keyed_prf_v1 verification")
        result = DualSlidingWindowVerifier(
            history_size=history_size,
            lookahead_window=check_window,
            length_tolerance=tolerated_errors,
        ).verify(detection_segs, manifest, master_key)
        if debug:
            print(result.status.value, result.diagnostics)
        return result.invalid_points
    if manifest is not None and manifest.schedule_version == "legacy_increment_v0":
        min_length = manifest.d_min
        max_length = manifest.d_max
        length_increment = manifest.length_step
        state_bit_size = manifest.state_bits
    elif manifest is not None and manifest.schedule_version != "keyed_prf_v1":
        raise ValueError(f"unsupported schedule version: {manifest.schedule_version}")

    def get_state_num(state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().tolist()
        return int(''.join(map(str, state)), 2)


    def next_expected_state(current_state):
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.cpu().tolist()
        binary_str = ''.join(map(str, current_state))
        current_num = int(binary_str, 2)
        next_num = (current_num + 1) % (2 ** state_bit_size)
        return [int(b) for b in bin(next_num)[2:].zfill(state_bit_size)]

    def next_expected_length(current_length):
        next_length = current_length + length_increment
        return next_length if next_length <= max_length else next_length - max_length + min_length

    invalid_points = []
    history_states = deque(maxlen=history_size)
    history_lengths = deque(maxlen=history_size)
    consecutive_errors = 0
    reset = True
    i = 0
    while i < len(detection_segs) - check_window:
        if i<10:
            loop_size=history_size
        else:
            loop_size=history_size+2

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


        start, end, state = detection_segs[i]
        current_length = end - start


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
                    times=state2+(2 ** state_bit_size)-state1+1
                expected_lengths.append(length + length_increment * times)





        valid_expected_lengths = [length for length in expected_lengths if length != -1]
        if len(valid_expected_lengths)==0:
            reset=True
            i+=1
            continue

        elif len(valid_expected_lengths) > 3:
            trimmed_lengths = sorted(valid_expected_lengths)[1:-1]
        else:
            trimmed_lengths = valid_expected_lengths

        expected_next_length = sum(trimmed_lengths) / len(trimmed_lengths)

        if expected_next_length > max_length:
            expected_next_length = expected_next_length - max_length + min_length


        expected_next_state = next_expected_state(history_states[-1])


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

        future_states = [expected_next_state]
        future_length=expected_next_length
        future_lengths = [future_length]
        for _ in range(check_window - 1):
            expected_next_state = next_expected_state(expected_next_state)
            future_length = next_expected_length(future_length)
            future_states.append(expected_next_state)

            future_lengths.append(future_length)


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


    invalid_points = sorted(invalid_points)

    return invalid_points
