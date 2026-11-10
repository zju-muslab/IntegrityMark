import torch
from torch.nn.functional import cosine_similarity

def compute_accuracy(positive, negative, mask=None):
    if positive.size(1)==0:
        return torch.tensor(0.0)

    if mask is not None:
        acc = ((positive>0) == (mask==1)).float().mean()
        return acc
    else:
        N = (positive[:, 0, :] > 0).float().mean() + (negative[:, 0, :] < 0).float().mean()
        acc = N / 2
        return acc

def calculate_tiou_by_sample(ground_truth, prediction):
    B, C, T = ground_truth.shape
    tious = []


    for c in range(C):
        class_tiou = []


        for b in range(B):
            gt = ground_truth[b,c]
            pred = prediction[b,c]


            if not gt.any() and not pred.any():
                continue


            intersection = (gt * pred).sum()
            union = torch.logical_or(gt, pred).sum()

            if union > 0:
                tiou = intersection.float() / union.float()
                class_tiou.append(tiou)


        if class_tiou:
            tious.append(torch.mean(torch.tensor(class_tiou)).item())


    return sum(tious)/len(tious), tious


def calculate_tiou(ground_truth, prediction):
    total_intersection = 0
    total_union = 0


    for gt in ground_truth:

        gt_start, gt_end, gt_state = gt


        for pred in prediction:
            pred_start, pred_end, pred_state = pred

            if torch.isclose(gt_state.float(), pred_state.float()).all() :

                intersection_start = max(gt_start, pred_start)
                intersection_end = min(gt_end, pred_end)

                if intersection_end > intersection_start:
                    intersection = intersection_end - intersection_start


                    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection

                    total_intersection += intersection
                    total_union += union

    if total_union == 0:
        return 0.0

    return total_intersection / total_union


class SDR_Evaluator:
    def __init__(self):
        self.sdr=0.0
        self.num_success=0
        self.num_segments=0


    def update(self, pred_scores, ground_truth_segments):

        num_segments = len(ground_truth_segments)
        if num_segments>100:
            num_segments=num_segments-100
            ground_truth_segments=ground_truth_segments[50:-50]

        if num_segments == 0:
            return 0.0

        num_success = 0
        pred_scores = pred_scores.squeeze(0)

        for start_idx, end_idx, state in ground_truth_segments:
            segment_preds = pred_scores[:, start_idx:end_idx]
            pred_states = segment_preds.argmax(dim=0)
            true_state = state.argmax().item()
            correct_ratio = (pred_states == true_state).float().mean()

            if correct_ratio > 0.1:
                num_success += 1

        self.num_success += num_success
        self.num_segments += num_segments
        return num_success/num_segments * 100

    def get_sdr(self):
        if self.num_segments == 0:
            return 0.0
        return (self.num_success / self.num_segments) * 100.0

    def reset(self):
        self.sdr = 0.0
        self.num_success = 0
        self.num_segments = 0


def calculate_tiou_state1(ground_truth, prediction, similarity_threshold=0.9):
    if not ground_truth or not prediction:
        return 0.0

    total_intersection = 0
    total_union = 0


    ground_truth = sorted(ground_truth, key=lambda x: x[0])
    prediction = sorted(prediction, key=lambda x: x[0])

    i, j = 0, 0
    while i < len(ground_truth) and j < len(prediction):
        gt_start, gt_end, gt_state = ground_truth[i]
        pred_start, pred_end, pred_state = prediction[j]


        if gt_state == 1:

            intersection_start = max(gt_start, pred_start)
            intersection_end = min(gt_end, pred_end)

            if intersection_end > intersection_start:

                if pred_state == 1:

                    intersection = intersection_end - intersection_start
                    total_intersection += intersection


                    union_start = min(gt_start, pred_start)
                    union_end = max(gt_end, pred_end)
                    total_union += union_end - union_start


        if gt_end < pred_end:
            i += 1
        else:
            j += 1


    if total_union == 0:
        return 0.0


    tiou = total_intersection / total_union
    return tiou

def calculate_tiou_fast(ground_truth, prediction, similarity_threshold=0.9, only_one=False):
    if not ground_truth or not prediction:
        return 0.0

    total_intersection = 0
    gt_coverage = set()
    pred_coverage = set()


    ground_truth = sorted(ground_truth, key=lambda x: x[0])
    prediction = sorted(prediction, key=lambda x: x[0])

    i, j = 0, 0
    while i < len(ground_truth) and j < len(prediction):

        gt_start, gt_end, gt_state = ground_truth[i]
        pred_start, pred_end, pred_state = prediction[j]


        assert isinstance(gt_start, int) and isinstance(gt_end, int), "Time points must be integers"


        if gt_end <= pred_start:
            gt_coverage.update(range(gt_start, gt_end))
            i += 1
            continue
        if pred_end <= gt_start:
            pred_coverage.update(range(pred_start, pred_end))
            j += 1
            continue


        if isinstance(gt_state, torch.Tensor):
            sim = cosine_similarity(gt_state.unsqueeze(0).float(),
                                pred_state.unsqueeze(0).float()).item()
        else:
            epsilon = 1e-6
            if abs(gt_state - pred_state) < epsilon:
                sim = 1.0
            else:
                sim = 0.0


        if sim >= similarity_threshold:
            intersection_start = max(gt_start, pred_start)
            intersection_end = min(gt_end, pred_end)
            if intersection_end > intersection_start:
                gt_coverage.update(range(gt_start, gt_end))
                pred_coverage.update(range(pred_start, pred_end))
                total_intersection += intersection_end - intersection_start
        else:

            gt_coverage.update(range(gt_start, gt_end))
            pred_coverage.update(range(pred_start, pred_end))


        if gt_end < pred_end:
            i += 1
        else:
            j += 1


    while i < len(ground_truth):
        gt_start, gt_end, _ = ground_truth[i]
        gt_coverage.update(range(gt_start, gt_end))
        i += 1

    while j < len(prediction):
        pred_start, pred_end, _ = prediction[j]
        pred_coverage.update(range(pred_start, pred_end))
        j += 1

    if not gt_coverage or not pred_coverage:
        return 0.0

    total_union = len(gt_coverage.union(pred_coverage))
    return total_intersection / total_union

def compute_bit_acc(decoded, message, mask=None, sigmoid=False):
    if decoded.size(1)==0:
        return torch.tensor(0.0)

    if mask is not None:

        new_shape = [*decoded.shape[:-1], -1]
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)
        message = torch.masked_select(message, mask == 1).reshape(new_shape)


    if sigmoid:
        decoded = decoded.sigmoid()
    decoded = decoded > 0.5
    res = (decoded == message).float().mean()
    return res
