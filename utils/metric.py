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
    """
    计算每个类别的TIoU并返回平均值
    
    Args:
        ground_truth: [B,C,T] tensor, 每个点是0或1
        prediction: [B,C,T] tensor, 每个点是0或1
    Returns:
        float: 平均TIoU
        List[float]: 每个类别的TIoU
    """
    B, C, T = ground_truth.shape
    tious = []
    
    # 对每个类别分别计算
    for c in range(C):
        class_tiou = []
        
        # 对每个batch分别计算
        for b in range(B):
            gt = ground_truth[b,c]    # [T]
            pred = prediction[b,c]     # [T]
            
            # 如果该类别在这个序列中没有出现，跳过
            if not gt.any() and not pred.any():
                continue
            
            # 计算intersection和union
            intersection = (gt * pred).sum()
            union = torch.logical_or(gt, pred).sum()
            
            if union > 0:
                tiou = intersection.float() / union.float()
                class_tiou.append(tiou)
        
        # 计算该类别在所有batch上的平均TIoU
        if class_tiou:
            tious.append(torch.mean(torch.tensor(class_tiou)).item())
    
    # 返回所有类别的平均值和每个类别的具体值
    return sum(tious)/len(tious), tious


def calculate_tiou(ground_truth, prediction):
    """
    计算时序IoU (Temporal IoU)
    
    Args:
        ground_truth: List of [start, end, state]
        prediction: List of [start, end, state]
    
    Returns:
        float: TIoU score
    """
    total_intersection = 0
    total_union = 0

    # 对每个groundtruth段落
    for gt in ground_truth:
        
        gt_start, gt_end, gt_state = gt
        
        # 找到对应state的预测段落
        for pred in prediction:
            pred_start, pred_end, pred_state = pred
            # 只比较相同state的段落
            if torch.isclose(gt_state.float(), pred_state.float()).all() :
                # 计算交集
                intersection_start = max(gt_start, pred_start)
                intersection_end = min(gt_end, pred_end)
                
                if intersection_end > intersection_start:
                    intersection = intersection_end - intersection_start
                    
                    # 计算并集
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
        """
        Update SDR statistics with new batch of predictions
        
        Args:
            pred_scores: torch.Tensor of prediction scores, shape [1, C, T]
            ground_truth_segments: list of tuples (start_idx, end_idx, state)
                                 where state is a tensor of shape [C]
        """

        num_segments = len(ground_truth_segments)
        if num_segments>100:
            num_segments=num_segments-100
            ground_truth_segments=ground_truth_segments[50:-50]

        if num_segments == 0:
            return 0.0
            
        num_success = 0
        pred_scores = pred_scores.squeeze(0)  # [C, T]
        
        for start_idx, end_idx, state in ground_truth_segments:
            segment_preds = pred_scores[:, start_idx:end_idx]  # [C, segment_length]
            pred_states = segment_preds.argmax(dim=0)  # [segment_length]
            true_state = state.argmax().item()
            correct_ratio = (pred_states == true_state).float().mean()
            
            if correct_ratio > 0.1:
                num_success += 1
                
        self.num_success += num_success
        self.num_segments += num_segments
        return num_success/num_segments * 100

    def get_sdr(self):
        """
        Get current overall SDR
        
        Returns:
            sdr: Current SDR value in percentage
        """
        if self.num_segments == 0:
            return 0.0
        return (self.num_success / self.num_segments) * 100.0

    def reset(self):
        """Reset accumulated statistics"""
        self.sdr = 0.0
        self.num_success = 0
        self.num_segments = 0


def calculate_tiou_state1(ground_truth, prediction, similarity_threshold=0.9):
    """
    计算 TIoU，仅考虑状态值为 1 且与 ground_truth 有重叠的 prediction 时间段。
    
    Args:
        ground_truth: List of tuples (start, end, state)
        prediction: List of tuples (start, end, state)
        similarity_threshold: 状态相似度阈值（用于状态值为张量的情况）
    
    Returns:
        float: TIoU 值
    """
    if not ground_truth or not prediction:  # 输入检查
        return 0.0

    total_intersection = 0
    total_union = 0

    # 确保已排序
    ground_truth = sorted(ground_truth, key=lambda x: x[0])
    prediction = sorted(prediction, key=lambda x: x[0])

    i, j = 0, 0
    while i < len(ground_truth) and j < len(prediction):
        gt_start, gt_end, gt_state = ground_truth[i]
        pred_start, pred_end, pred_state = prediction[j]

        # 仅处理 ground_truth 中状态值为 1 的时间段
        if gt_state == 1:
            # 检查 prediction 是否与当前 ground_truth 有重叠
            intersection_start = max(gt_start, pred_start)
            intersection_end = min(gt_end, pred_end)

            if intersection_end > intersection_start:  # 有重叠
                # 检查 prediction 的状态值是否为 1
                if pred_state == 1:
                    # 计算交集
                    intersection = intersection_end - intersection_start
                    total_intersection += intersection

                    # 计算并集
                    union_start = min(gt_start, pred_start)
                    union_end = max(gt_end, pred_end)
                    total_union += union_end - union_start

        # 移动指针
        if gt_end < pred_end:
            i += 1
        else:
            j += 1

    # 如果没有重叠或没有状态值为 1 的时间段
    if total_union == 0:
        return 0.0

    # 计算 TIoU
    tiou = total_intersection / total_union
    return tiou

def calculate_tiou_fast(ground_truth, prediction, similarity_threshold=0.9, only_one=False):
    """
    计算TIoU，考虑状态相似度
    Args:
        ground_truth: List of tuples (start, end, state)
        prediction: List of tuples (start, end, state)
        similarity_threshold: 状态相似度阈值
    Returns:
        float: TIoU值
    """
    if not ground_truth or not prediction:  # 添加输入检查
        return 0.0
        
    total_intersection = 0
    gt_coverage = set()
    pred_coverage = set()

    # 确保已排序
    ground_truth = sorted(ground_truth, key=lambda x: x[0])
    prediction = sorted(prediction, key=lambda x: x[0])

    i, j = 0, 0
    while i < len(ground_truth) and j < len(prediction):

        gt_start, gt_end, gt_state = ground_truth[i]
        pred_start, pred_end, pred_state = prediction[j]

        # 添加类型检查
        assert isinstance(gt_start, int) and isinstance(gt_end, int), "Time points must be integers"
        
        # 不重叠的情况
        if gt_end <= pred_start:
            gt_coverage.update(range(gt_start, gt_end))
            i += 1
            continue
        if pred_end <= gt_start:
            pred_coverage.update(range(pred_start, pred_end))
            j += 1
            continue

        # 计算状态相似度
        if isinstance(gt_state, torch.Tensor):
            sim = cosine_similarity(gt_state.unsqueeze(0).float(), 
                                pred_state.unsqueeze(0).float()).item()
        else: # 数值
            epsilon = 1e-6  # 定义一个小的误差范围
            if abs(gt_state - pred_state) < epsilon:
                sim = 1.0  # 如果数值在误差范围内，视为相等
            else:
                sim = 0.0  # 否则视为不相等
                    
        # 重叠且状态相似
        if sim >= similarity_threshold:
            intersection_start = max(gt_start, pred_start)
            intersection_end = min(gt_end, pred_end)
            if intersection_end > intersection_start:
                gt_coverage.update(range(gt_start, gt_end))
                pred_coverage.update(range(pred_start, pred_end))
                total_intersection += intersection_end - intersection_start
        else:
            # 即使状态不相似也需要更新coverage
            gt_coverage.update(range(gt_start, gt_end))
            pred_coverage.update(range(pred_start, pred_end))

        # 移动指针
        if gt_end < pred_end:
            i += 1
        else:
            j += 1

    # 处理剩余的序列
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
        # cut last dim of positive to keep only where mask is 1
        new_shape = [*decoded.shape[:-1], -1]  # b nbits t -> b nbits -1
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)
        message = torch.masked_select(message, mask == 1).reshape(new_shape)
    # average decision over time, then threshold
    # decoded = decoded.mean(dim=-1) > 0  # b nbits
    if sigmoid:
        decoded = decoded.sigmoid()
    decoded = decoded > 0.5  # b nbits
    res = (decoded == message).float().mean()
    return res