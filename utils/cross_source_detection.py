def evaluate_tampering_overlap_ratio(detection_segs, tamper_segs):

    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]


    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1 and seg[1] - seg[0] >= 8000]


    if not detection_segs_1 and not tamper_segs_1:
        return 1.0, 1.0, 1.0, 0.0

    if not detection_segs_1:
        return 0.0, 0.0, 0.0, float('inf')

    if not tamper_segs_1:
        return 0.0, 1.0, 0.0, float('inf')


    total_detected_length = sum(seg[1] - seg[0] for seg in detection_segs_1)


    total_tamper_length = sum(seg[1] - seg[0] for seg in tamper_segs_1)


    total_overlap_length = 0


    for detect_seg in detection_segs_1:
        detect_start, detect_end = detect_seg[0], detect_seg[1]

        for tamper_seg in tamper_segs_1:
            tamper_start, tamper_end = tamper_seg[0], tamper_seg[1]


            overlap_start = max(detect_start, tamper_start)
            overlap_end = min(detect_end, tamper_end)


            if overlap_start < overlap_end:
                total_overlap_length += (overlap_end - overlap_start)



    precision = total_overlap_length / total_detected_length if total_detected_length > 0 else 0


    recall = total_overlap_length / total_tamper_length if total_tamper_length > 0 else 0


    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


    toe_values = []


    for detect_seg in detection_segs_1:
        detect_start, detect_end = detect_seg[0], detect_seg[1]


        min_left_error = float('inf')
        for tamper_seg in tamper_segs_1:
            tamper_start = tamper_seg[0]
            error = abs(detect_start - tamper_start)
            if error < min_left_error:
                min_left_error = error


        min_right_error = float('inf')
        for tamper_seg in tamper_segs_1:
            tamper_end = tamper_seg[1]
            error = abs(detect_end - tamper_end)
            if error < min_right_error:
                min_right_error = error


        if min_left_error != float('inf'):
            toe_values.append(min_left_error)
        if min_right_error != float('inf'):
            toe_values.append(min_right_error)


    toe = sum(toe_values) / len(toe_values) if toe_values else float('inf')

    return precision, recall, f1, toe






def evaluate_tampering_boundary_accuracy(detection_segs, tamper_segs):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    le = []


    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]


    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1 and seg[1] - seg[0] >= 8000]


    tp_detection_indices = set()


    for tamper_seg in tamper_segs_1:
        found_match = False
        min_error = float('inf')
        best_index = -1
        best_head_error = float('inf')
        best_tail_error = float('inf')


        for i, detect_seg in enumerate(detection_segs_1):

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


        if found_match and best_head_error < 1600 and best_tail_error < 1600:
            TP += 1
            tp_detection_indices.add(best_index)
            le.append(best_head_error)
            le.append(best_tail_error)
        else:
            FN += 1


    FP = len(detection_segs_1) - len(tp_detection_indices)


    if not detection_segs_1 and not tamper_segs_1:
        TN = 1

    return TP, FN, TN, FP, le
