import torch
import torch.nn.functional as F


class MsgGenerator():
    """
    水印消息生成器 - 根据配置生成三种类型的水印消息

    基于论文 "IntegrityMark: Tamper-Resistant Audio Watermarking for Real-Time Integrity Protection"

    三种水印消息类型：
    1. Fix (固定水印): 全局静态标识，整个音频使用相同值，用于检测跨源篡改
    2. Temp (时序水印): 按固定间隔变化的时序标识，提供粗粒度的时间戳
    3. Val (可变长度验证水印): 论文核心创新，通过状态序列+段长度的双重验证检测同源篡改

    三种消息的组合方式由配置文件指定，如 "0+2+4" 表示：
    - 0 bit fix水印 + 2 bit temp水印 + 4 bit val水印 = 6 bits总消息
    """

    def __init__(self, msg_cfg):
        """
        初始化消息生成器

        Args:
            msg_cfg: 配置对象，包含 fix/temp/val 三种消息的参数
                - fix.length: 固定水印的比特数
                - temp.length: 时序水印的比特数
                - temp.interval: 时序水印的更新间隔（采样点）
                - val.length: 可变长度水印的比特数 (V值)
                - val.d_min: 最小段长度（采样点）
                - val.d_max: 最大段长度（采样点）
                - val.increment: 段长递增步长（采样点）
        """
        # ========== Fix (固定水印) 参数 ==========
        self.msg_fix_length = msg_cfg.fix.length

        # ========== Temp (时序水印) 参数 ==========
        self.msg_temp_length = msg_cfg.temp.length
        self.msg_temp_interval = msg_cfg.temp.interval
        self.msg_temp_interval_range = msg_cfg.temp.interval_range

        # ========== Val (可变长度验证水印) 参数 - 论文核心 ==========
        self.msg_val_length = msg_cfg.val.length  # V值，决定了可能的状态数: 2^V
        self.msg_val_interval = msg_cfg.val.interval
        self.msg_val_interval_range = msg_cfg.val.interval_range

        # 检查是否使用论文中的可变长度方案 (d_min, d_max, increment)
        if 'd_min' in msg_cfg.val:
            self.use_d_min_max = True
            self.d_min = msg_cfg.val.d_min          # 最小段长度
            self.d_max = msg_cfg.val.d_max          # 最大段长度
            self.val_increment = msg_cfg.val.increment  # 递增步长

        # 缓存生成的消息
        self.msg_fix = None
        self.msg_temp = None
        self.msg_val = None
        self.msg_total = None

    def msg_generate(self, x, return_seg=False):
        """
        生成完整的水印消息集合

        将 fix、temp、val 三种消息按顺序拼接成一个统一的消息张量，
        供水印嵌入器使用。

        Args:
            x: 输入音频张量，shape: [batch_size, channels, length]
            return_seg: 是否返回分段信息（用于调试和篡改定位）

        Returns:
            msg_total: 拼接后的所有消息，shape: [batch_size, (fix_len+temp_len+val_len), length]
            msg_fix: 固定水印消息，shape: [batch_size, fix_len, length] 或 None
            msg_temp: 时序水印消息，shape: [batch_size, temp_len, length] 或 None
            msg_val: 验证水印消息，shape: [batch_size, val_len, length] 或 None
            msg_seg: (可选) 分段信息列表，用于追踪每个消息的位置
        """
        b, _, t = x.shape

        # 初始化消息张量，按顺序拼接三种消息
        self.msg_total = torch.zeros(b, 0, t).to(x.device)
        self.msg_seg = []

        # ========== 1. 生成 Fix 消息 (固定全局标识) ==========
        if self.msg_fix_length > 0:
            # 每个batch生成随机的fix消息（整个音频使用相同值）
            self.msg_fix = self.msg_fix_random_generate(batch_size=b, msg_length=self.msg_fix_length)
            # 在时间维度重复，使消息覆盖整个音频
            self.msg_fix = self.msg_fix.repeat(1, 1, t)
            self.msg_fix = self.msg_fix.to(x.device)
            self.msg_total = torch.cat((self.msg_total, self.msg_fix), dim=1)

        # ========== 2. 生成 Temp 消息 (时序标识) ==========
        msg_temp_seg = []
        if self.msg_temp_length > 0:
            # 使用随机消息生成方案：每隔interval采样点更换一次消息
            self.msg_temp, msg_temp_seg = self.random_rec_msg_generate(
                x,
                temp_msg_length=self.msg_temp_length,
                interval=self.msg_temp_interval,
                interval_range=self.msg_temp_interval_range
            )
            self.msg_temp = self.msg_temp.to(x.device)
            self.msg_total = torch.cat((self.msg_total, self.msg_temp), dim=1)
        self.msg_seg.append(msg_temp_seg)

        # ========== 3. 生成 Val 消息 (可变长度验证 - 论文核心) ==========
        msg_val_seg = []
        if self.msg_val_length > 0:
            if self.use_d_min_max:
                # 使用论文推荐的可变长度方案
                # 将状态序列以变长段的形式嵌入，用于双重验证
                self.msg_val, msg_val_seg = self.val_msg_generate(
                    x,
                    val_msg_bits=self.msg_val_length,
                    d_min=self.d_min,
                    d_max=self.d_max,
                    increment=self.val_increment
                )
            else:
                # 降级方案：使用固定或随机间隔
                self.msg_val, msg_val_seg = self.circle_rec_msg_generate(
                    x,
                    temp_msg_length=self.msg_val_length,
                    interval=self.msg_val_interval,
                    interval_range=self.msg_val_interval_range
                )
            self.msg_val = self.msg_val.to(x.device)
            self.msg_total = torch.cat((self.msg_total, self.msg_val), dim=1)
        self.msg_seg.append(msg_val_seg)

        # 返回结果
        if return_seg:
            return self.msg_total, self.msg_fix, self.msg_temp, self.msg_val, self.msg_seg
        return self.msg_total, self.msg_fix, self.msg_temp, self.msg_val

    def msg_fix_random_generate(self, batch_size, msg_length):
        """
        生成固定水印消息 - 全局静态标识

        为每个batch生成一个随机的msg_length位的消息，
        该消息将被重复覆盖整个音频长度。

        用途：
        - 标记音频来源/所有者
        - 标识同一直播会话
        - 检测跨源篡改（当外部内容被插入时，缺少此消息）

        Args:
            batch_size: 批次大小
            msg_length: 消息位数（如fix.length=0则不生成）

        Returns:
            msg: 形状 [batch_size, msg_length, 1] 的二进制消息
        """
        msg = torch.randint(0, 2, (batch_size, msg_length, 1), dtype=torch.float32)
        return msg 
    
    def temp_msg_generate(self, x, temp_msg_length,interval=128):
        batch_size = x.shape[0]
        sequences_list = []
        for _ in range(batch_size):
            n = torch.randint(2**temp_msg_length, (1,)).item()
            sequences = torch.tensor([[int(x) for x in f"{i:04b}"] for i in range(2**temp_msg_length)])
            sequences = torch.roll(sequences, shifts=n, dims=0)
            repeated_sequences = sequences.repeat_interleave(interval, dim=0)
            while repeated_sequences.shape[0] < x.shape[-1]:
                repeated_sequences = torch.cat((repeated_sequences, sequences.repeat_interleave(interval, dim=0)), dim=0)
            repeated_sequences = repeated_sequences[:x.shape[-1]]
            sequences_list.append(repeated_sequences)

        sequences = torch.stack(sequences_list).permute(0,2,1)
        return sequences

    def circle_rec_msg_generate(self, x, temp_msg_length, interval=128, interval_range=[0.5,4.0]):
        batch_size = x.shape[0]
        length = x.shape[-1]

        repeated_sequences = torch.zeros(batch_size, temp_msg_length, length)
        msg_seg=[]
        for i in range(batch_size):
            n = torch.randint(2**temp_msg_length, (1,)).item()

            sequences = torch.tensor([[int(x) for x in f"{i:0{temp_msg_length}b}"] for i in range(2**temp_msg_length)])
            sequences = torch.roll(sequences, shifts=n, dims=0)
            curr=0
            while curr < length:
                for seq in sequences:
                    if interval_range is False:
                        real_interval = torch.randint(1, 6, (1,)).item()*interval
                    else:
                        real_interval = min(torch.randint(int(interval * interval_range[0]), int(interval * interval_range[1]), (1,)).item(),length-curr)
                    if real_interval+curr>length:
                        real_interval=length-curr
                    # print(repeated_sequences.shape,seq.reshape(1,-1,1).repeat(1, 1, real_interval).shape)
                    msg_seg.append([curr,curr+real_interval,seq.reshape(-1)])
                    repeated_sequences[i, :, curr:curr+real_interval] += seq.reshape(-1,1)
                    curr=curr+real_interval
                    if curr>=length:
                        break
        return repeated_sequences, msg_seg


    def random_rec_msg_generate(self, x, temp_msg_length, interval=128, interval_range=[0.5,4.0]):
        batch_size = x.shape[0]
        length = x.shape[-1]

        repeated_sequences = torch.zeros(batch_size, temp_msg_length, length)
        msg_seg=[]
        for i in range(batch_size):
            curr=0
            while curr < length:
                if interval==-1:
                    real_interval=length
                elif interval_range is False:
                    real_interval=interval
                else:
                    real_interval = min(torch.randint(int(interval * interval_range[0]), int(interval * interval_range[1]), (1,)).item(),length-curr)
                if real_interval+curr>length:
                    real_interval=length-curr
                msg = torch.randint(0, 2, (temp_msg_length, 1), dtype=torch.float32) # b, n, t=1
                msg_seg.append([curr,curr+real_interval,msg.reshape(-1)])
                repeated_sequences[i, :, curr:curr+real_interval] += msg
                curr=curr+real_interval
        return repeated_sequences, msg_seg


    def val_msg_generate(self, x, val_msg_bits, d_min, d_max, increment=32):
        """
        生成论文核心的可变长度验证水印 (Varying-Length Validation Watermark)

        这是 IntegrityMark 论文的主要创新，通过两个特征的双重验证来检测同源篡改：
        1. 状态序列验证：状态按 00→01→10→11→00... 循环递增
        2. 段长度验证：段长按 d_min → d_min+n → ... → d_max → d_min 递增

        工作原理：
        - 生成 2^V 种可能的V位状态（如V=4则16种状态：0000,0001,...,1111）
        - 随机打乱这些状态的顺序（增加安全性）
        - 将每个状态嵌入到可变长度的音频段中
        - 段长度按固定步长递增，到达最大值后循环回到最小值

        篡改检测机制：
        - 若攻击者删除/插入/替换音频片段：
          * 状态序列会出现跳变（不符合预期的循环模式）
          * 段长度模式会断裂（不符合递增规律）
        - 双重检查确保任何篡改都会被检测

        论文参考：Section 4.3, Figure 4, Algorithm 1

        Args:
            x: 输入音频张量，shape: [batch_size, channels, length]
            val_msg_bits: V值（验证消息的比特数），决定了状态数: 2^V
                         如V=4，则有16种状态 (0000, 0001, ..., 1111)
            d_min: 最小段长度（采样点），如512个采样点 (~32ms@16kHz)
            d_max: 最大段长度（采样点），如2064个采样点 (~128ms@16kHz)
            increment: 段长递增步长（采样点），如32个采样点

        Returns:
            repeated_sequences: 生成的验证消息，shape: [batch_size, val_msg_bits, length]
                              包含val_msg_bits维的二进制向量，时间对齐到音频长度
            msg_segments: 分段信息列表，格式 [[start_pos, end_pos, state_vector], ...]
                         用于篡改定位和双重验证算法
        """
        batch_size = x.shape[0]
        length = x.shape[-1]

        # 初始化输出数组
        msg_length = 2 ** val_msg_bits  # 状态总数：2^V
        repeated_sequences = torch.zeros((batch_size, val_msg_bits, length))
        msg_segments = []

        # ========== 为每个样本生成可变长度验证水印 ==========
        for i in range(batch_size):
            # 生成循环递增的状态序列：00→01→10→11→00...
            shift = torch.randint(2 ** val_msg_bits, (1,)).item()
            # 生成所有可能的V位状态（二进制向量）
            sequences = torch.tensor(
                [[int(x) for x in f"{i:0{val_msg_bits}b}"] for i in range(2 ** val_msg_bits)]
            )
            # 随机打乱状态顺序（增加安全性，防止模式识别）
            sequences = torch.roll(sequences, shifts=shift, dims=0)

            # ========== 为该样本的整个音频生成变长分段 ==========
            curr_pos = 0
            curr_length = d_min

            while curr_pos < length:
                # 遍历所有状态，依次分配到变长的音频段
                for seq in sequences:
                    # 根据递增规律计算当前段的长度
                    curr_length += increment
                    # 当长度超过最大值时，循环回到最小值（形成周期）
                    if curr_length >= d_max:
                        curr_length = curr_length - d_max + d_min

                    # 确保不超过音频总长度
                    if curr_pos + curr_length > length:
                        curr_length = length - curr_pos

                    # 将该状态向量嵌入到当前段 [curr_pos, curr_pos+curr_length)
                    repeated_sequences[i, :, curr_pos:curr_pos + curr_length] += seq.reshape(-1, 1)
                    # 记录分段信息，用于后续的完整性验证
                    msg_segments.append([curr_pos, curr_pos + curr_length, seq.reshape(-1)])
                    curr_pos += curr_length

                    if curr_pos == length:
                        break
                if curr_pos == length:
                    break

        return repeated_sequences, msg_segments


def post_process(x, window_size=5, threshold=0.1, min_duration=128, return_seg=False):
    """
    水印消息后处理函数 - 从检测器输出转换到离散消息

    这是论文 Section 4.1 "Message Post-processing" 的实现。
    功能：
    1. 将连续的置信度分数 (logits) 转换为离散的二进制消息
    2. 平滑序列以降低噪声影响
    3. 过滤低置信度点（接近0.5的点）
    4. 合并短暂的状态变化（低于min_duration的状态被合并）

    工作流程：
    - sigmoid: 将logits转换为[0,1]的置信度
    - 滑动窗口平滑: 减少高频噪声
    - 阈值修正: 将接近0或1的值锐化，接近0.5的值保留为中间值
    - 二值化: 用0.5阈值转换为0/1
    - (可选) 分段合并: 合并过短的状态段

    Args:
        x: 检测器输出logits，shape: [batch_size, num_bits, time]
           - num_bits 通常为 1 (存在性) + val_bits + temp_bits + fix_bits
           - 如 0+2+4 配置则为 1+4+2+0=7 bits
        window_size: 滑动窗口大小（用于平滑）
        threshold: 置信度过滤阈值，默认0.1
                  高于 1-threshold (0.9) → 判定为1
                  低于 threshold (0.1) → 判定为0
                  [threshold, 1-threshold] → 保留原值
        min_duration: 最小状态持续时间（采样点）
                     短于此时长的状态段会被合并到相邻段
        return_seg: 是否返回分段信息（用于篡改定位）

    Returns:
        output: 离散化的二进制消息矩阵，shape: [batch_size, num_bits, time]
                值为0或1（或有return_seg时保留部分浮点值）
        segments: (可选) 分段信息，用于完整性验证算法
    """
    # ========== 第1步：置信度转换 ==========
    x = F.sigmoid(x)

    B, num_states, T = x.shape
    output = torch.zeros_like(x, dtype=torch.long)
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.to(x.device)

    # ========== 第2步：逐比特处理 ==========
    for state_idx in range(num_states):
        # 提取当前比特的置信度序列
        p = x[:, state_idx, :]

        # 滑动窗口平滑（1D卷积）- 减少噪声
        p_smoothed = F.conv1d(p, kernel.view(1, 1, -1), padding=window_size // 2)

        # 阈值修正和锐化
        # - 高置信度（>0.9）→ 1
        # - 低置信度（<0.1）→ 0
        # - 中等置信度（0.1-0.9）→ 保留平滑值
        p_corrected = torch.where(
            p_smoothed > 1 - threshold,
            torch.tensor(1.0, device=x.device),
            torch.where(
                p_smoothed < threshold,
                torch.tensor(0.0, device=x.device),
                p_smoothed
            )
        )

        # 二值化处理 - 最终转换为0或1
        output[:, state_idx, :] = (p_corrected > 0.5).long()

    if return_seg:
        # 获取初始分段
        segments = sequence_to_segments_fast(output)
        # 对每个状态的分段进行处理
        for batch_idx in range(B):
            batch_segment = segments[batch_idx]
            i = 0
            while i < len(batch_segment):
                curr_seg = batch_segment[i]
                duration = curr_seg[1] - curr_seg[0]
                
                if duration < min_duration:
                    # 找到前后相邻段
                    prev_seg = batch_segment[i-1] if i > 0 else None
                    next_seg = batch_segment[i+1] if i < len(batch_segment)-1 else None
                    
                    # 根据不同情况处理合并
                    if prev_seg is not None and next_seg is not None:
                        # 有前后两段
                        if torch.equal(prev_seg[2],next_seg[2]):  # 前后段状态相同
                            # 合并三段
                            new_seg = (prev_seg[0], next_seg[1], prev_seg[2])
                            batch_segment[i-1] = new_seg
                            batch_segment.pop(i)  # 删除当前段
                            batch_segment.pop(i)  # 删除后一段
                            i -= 1  # 回退一步，因为删除了当前段
                        else:  # 前后段状态不同
                            # 与前段合并
                            new_seg = (prev_seg[0], curr_seg[1], prev_seg[2])
                            batch_segment[i-1] = new_seg
                            batch_segment.pop(i)  # 删除当前段
                            i -= 1
                    elif prev_seg is not None:
                        # 只有前段，与前段合并
                        new_seg = (prev_seg[0], curr_seg[1], prev_seg[2])
                        batch_segment[i-1] = new_seg
                        batch_segment.pop(i)
                        i -= 1
                    elif next_seg is not None:
                        # 只有后段，与后段合并
                        new_seg = (curr_seg[0], next_seg[1], next_seg[2])
                        batch_segment[i] = new_seg
                        batch_segment.pop(i+1)
                i += 1
            segments[batch_idx] = batch_segment
            # # 更新处理后的分段
            # segments[state_idx] = batch_segment
            # # 根据更新后的分段重建output
            # for state_idx in range(num_states):
            #     output[batch_idx, state_idx, :] = 0
            #     for start, end, value in batch_segment:
            #         if value == 1:
            #             output[batch_idx, state_idx, start:end] = 1
        
        return output, segments
    else:
        return output

# def post_process(x, window_size=5, threshold=0.1, return_seg=False):
#     """
#     后处理函数：平滑序列并过滤小概率事件
#     Args:
#         x: 输入张量，形状为 [1, 6, T]，每个状态位的概率分布
#         window_size: 滑动窗口大小，用于平滑概率值
#         threshold: 概率过滤阈值，小于该值的状态将被修正
#     Returns:
#         output: 处理后的二进制矩阵，形状为 [1, 6, T]
#     """
#     x=F.sigmoid(x)
#     B, num_states, T = x.shape  # B = 1, num_states = 6, T = 时间步数
#     # 初始化输出张量
#     output = torch.zeros_like(x, dtype=torch.long)  # [1, 6, T]

#     # 对每个状态位单独处理
#     for state_idx in range(num_states):
#         # 提取当前状态位的概率序列，形状为 [1, T]
#         p = x[:, state_idx, :]  # [1, T]

#         # 滑动窗口平滑 (1D 卷积)
#         kernel = torch.ones(window_size) / window_size
#         kernel = kernel.to(x.device)  # 确保与输入设备一致
#         p_smoothed = F.conv1d(
#             p,  # 输入形状 [1, T]
#             kernel.view(1, 1, -1),  # 卷积核形状 [1, 1, window_size]
#             padding=window_size // 2
#         )  # 输出形状为 [1, T]

#         # 修正小概率事件：将低于阈值的概率修正为 0 或 1
#         p_corrected = torch.where(
#             p_smoothed > 1 - threshold,  # 如果接近 1
#             torch.tensor(1.0, device=x.device),
#             torch.where(p_smoothed < threshold,  # 如果接近 0
#                         torch.tensor(0.0, device=x.device),
#                         p_smoothed)  # 否则保留平滑值
#         )

#         # 二值化处理：最终输出为 0 或 1
#         output[:, state_idx, :] = (p_corrected > 0.5).long()

#     if return_seg:
#         segments=sequence_to_segments_fast(output)
#         return output, segments
#     else:
#         return output

def sequence_to_segments_fast(output):
    """
    将离散的消息序列转换为分段表示 - 用于篡改检测

    将shape为[B,C,T]的0/1序列转换为段落形式[start,end,state]，
    用于论文的完整性验证算法（Algorithm 1）中的双重检查。

    分段表示的优势：
    - 紧凑：只记录状态变化的位置
    - 高效：算法1直接在分段上工作
    - 直观：便于识别篡改（缺失段、异常长度等）

    例如输入 [0,0,0,1,1,0,0,1,1,1] 会转换为：
    - [0, 2, 0] (位置0-2状态为0)
    - [3, 4, 1] (位置3-4状态为1)
    - [5, 6, 0] (位置5-6状态为0)
    - [7, 9, 1] (位置7-9状态为1)

    Args:
        output: shape为[B,C,T]的张量，值为0或1
                - B: batch size
                - C: 消息比特数
                - T: 时间步数

    Returns:
        segments: 列表的列表，格式：[[[start, end, state], ...], ...]
                 每个样本一个子列表，每个子列表包含该样本的所有段落
                 分段格式：[start_position, end_position, state_tensor]
    """
    B, C, T = output.shape
    segments = []
    
    # 将output转换为[B,T,C]以便于比较相邻时间步
    output = output.permute(0, 2, 1)  # [B,T,C]
    
    for b in range(B):
        # 找到状态改变的位置
        state_changes = (output[b, 1:] != output[b, :-1]).any(dim=1)  # [T-1]
        change_indices = torch.where(state_changes)[0]
        
        # 添加开始和结束索引
        all_indices = torch.cat([
            torch.tensor([0], device=output.device),
            change_indices + 1,
            torch.tensor([T], device=output.device)
        ])
        
        # 构建段落
        batch_segments = []
        for i in range(len(all_indices) - 1):
            start = all_indices[i].item()
            end = all_indices[i + 1].item() - 1
            if i == len(all_indices) - 2:  # 最后一段
                end = T - 1
            state = output[b, start]
            batch_segments.append([start, end, state.detach()])
            
        segments.append(batch_segments)
    
    return segments

    
    return segments

def sequence_to_segments(output):
    """
    将shape为[B,C,T]的0/1序列转换为段落形式[start,end,state]
    state为C维度的向量
    
    Args:
        output: shape为[B,C,T]的张量,值为0或1
        
    Returns:
        segments: 列表的列表,每个子列表包含该batch的所有段落
                 段落格式为[start,end,state_vector]
    """
    B, C, T = output.shape
    segments = []
    
    for b in range(B):
        batch_segments = []
        # 获取当前时刻的C维状态向量
        current_state = output[b, :, 0]  # shape: [C]
        start = 0
        
        for t in range(1, T):
            state_t = output[b, :, t]  # 当前时刻的状态向量
            # 如果状态发生变化
            if not (state_t == current_state).all():
                # 记录之前的段落
                batch_segments.append([start, t-1, current_state.clone()])
                # 更新信息
                start = t
                current_state = state_t
        
        # 处理最后一段
        batch_segments.append([start, T-1, current_state.clone()])
        segments.append(batch_segments)
    
    return segments



def crop(signal, watermark, cfg, msg_tmp=None, clip=False, return_seg=False):
    """
    训练数据增强函数 - 模拟音频篡改攻击

    在训练过程中，为了增强水印的鲁棒性和篡改检测能力，对音频进行各种变换模拟攻击。
    这实现了论文 Section 4.1 "Data Augmentation" 中的三种增强策略：

    1. Watermark Masking（水印掩码）:
       - 创建交替的水印和非水印区域
       - 教会模型准确检测水印边界
       - 模拟篡改时可能出现的不完整水印覆盖

    2. Tampering Simulation（篡改模拟）:
       - 删除和插入操作：模拟同源/跨源篡改
       - 随机替换音频片段：增加检测难度
       - 可变长度的操作：防止模型过拟合固定的攻击模式

    3. 跨源/同源篡改：
       - 跨源（Cross-source）：用批次中其他样本或噪声替换
       - 同源（In-source）：用同一音频的其他片段替换

    参数配置（来自 config 文件的 crop 部分）:
       - prob: 裁剪操作的概率
       - shuffle_prob: 随机打乱的概率
       - pad_prob: 填充（掩码）的概率
       - insert_prob: 插入的概率
       - size: 操作的大小比例（相对于总长度）
       - max_n_windows: 最大窗口数
       - insert_size: 插入片段的长度范围 [min, max]

    Args:
        signal (torch.Tensor): 原始音频信号，shape: [batch_size, channels, length]
        watermark (torch.Tensor): 水印信号，shape: [batch_size, 1, length]
        cfg: 配置对象，包含增强参数（prob, shuffle_prob, pad_prob, insert_prob 等）
        msg_tmp (torch.Tensor): 水印消息，shape: [batch_size, num_bits, length]，用于篡改模拟
        clip (bool): 未使用的参数
        return_seg (bool): 未使用的参数

    Returns:
        signal (torch.Tensor): 修改后的音频信号
        watermark (torch.Tensor): 修改后的水印信号（应用了掩码）
        mask (torch.Tensor): 掩码张量，1表示有水印，0表示无水印
                           shape: [batch_size, 1, length]
        msg_tmp (torch.Tensor): 修改后的消息（如果输入不为None）
    """
    assert (
        cfg.prob + cfg.shuffle_prob + cfg.pad_prob
        <= 1
    ), f"The sum of the probabilities {cfg.prob=} {cfg.shuffle_prob=} \
            {cfg.pad_prob=} should be less than 1"
    # signal_cloned = signal.clone().detach()  # detach to be sure
    # watermark = watermark.clone().detach()  # detach to be sure
    mask = torch.ones_like(watermark)
    p = torch.rand(1)
    if p < cfg.pad_prob:  # Pad with some probability
        start = int(torch.rand(1) * 0.33 * watermark.size(-1))
        finish = int((0.66 + torch.rand(1) * 0.33) * watermark.size(-1))
        mask[:, :, :start] = 0
        mask[:, :, finish:] = 0
        if torch.rand(1) > 0.5:
            mask = 1 - mask
        signal *= mask  # pad signal

        # if msg_tmp is not None:
        #     msg_tmp *= mask  # pad msg
    
    elif (
        p < cfg.prob + cfg.pad_prob + cfg.shuffle_prob
    ):
        # Define a mask, then crop or shuffle
        mask_size = round(watermark.shape[-1] * cfg.size)
        n_windows = int(
            torch.randint(1, cfg.max_n_windows + 1, (1,)).item()
        )
        window_size = int(mask_size / n_windows)
        for _ in range(n_windows):  # Create multiple windows in the mask
            mask_start = torch.randint(0, watermark.shape[-1] - window_size, (1,))
            mask[:, :, mask_start: mask_start + window_size] = (
                0  # Apply window to mask
            )
        # inverse the mask half the time
        if torch.rand(1) > 0.5:
            mask = 1 - mask

        if p < cfg.pad_prob + cfg.shuffle_prob:  # shuffle
            # shuffle
            if signal.shape[0]>1:  
                shuffle_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                while (shuffle_idx == torch.arange(signal.size(0))).any():  # 检查是否有自身索引
                    shuffle_idx = torch.randint(0, signal.size(0), (signal.size(0),))
            else:
                shuffle_idx=0
            # signal_cloned=signal_cloned.repeat(1,1,2)
            # signal_cloned=signal_cloned[:,:,16000:signal.shape[-1]+16000]
            
            signal = signal * mask + signal[shuffle_idx] * (1 - mask)  # shuffle signal where not wm
            
            # if msg_tmp is not None:
            #     msg_cloned = msg_tmp.clone().detach()  # detach to be sure
            #     msg_tmp = msg_tmp * mask + msg_cloned[shuffle_idx] * (1 - mask)  # shuffle msg where not wm

    elif p < cfg.prob + cfg.pad_prob + cfg.shuffle_prob + cfg.insert_prob:
        insert_num=int(torch.randint(1, cfg.max_n_windows, (1,)).item())
        for _ in range(insert_num):
            
            insert_len_min = int(cfg.insert_size[0])
            insert_len_max = int(cfg.insert_size[1])
            insert_start = int(0.11 * watermark.shape[-1])  # Start position (1/3 of the signal)
            insert_end = int(0.88 * watermark.shape[-1])  # End position (2/3 of the signal)
            insert_length = torch.randint(insert_len_min, insert_len_max, (1,))
            insert_pos = torch.randint(insert_start, insert_end, (1,)).item()
            
            if insert_start + insert_length > signal.shape[-1]:
                insert_length=signal.shape[-1]-insert_start
            insert_length=insert_length-insert_length%100
            
            
        # # Insert new segments from other signals in the batch
        # insert_len_min = int(cfg.insert_size[0])
        # insert_len_max = int(cfg.insert_size[1])
        # insert_length = torch.randint(insert_len_min, insert_len_max, (1,))
        # insert_start = int(0.1 * watermark.shape[-1])  # Start position (1/3 of the signal)
        # insert_end = int(0.66 * watermark.shape[-1])  # End position (2/3 of the signal)
        # insert_pos = torch.randint(insert_start, insert_end, (1,)).item()
        # if insert_start + insert_length > signal_cloned.shape[-1]:
        #     insert_length=signal_cloned.shape[-1]-insert_start

            if signal.size(0) > 1:
                # Randomly select another signal from the batch
                insert_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                while (insert_idx == torch.arange(signal.size(0))).any():  # 检查是否有自身索引
                    insert_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                insert_segment = signal[insert_idx, :, insert_start:insert_start + insert_length]
                insert_watermark = watermark[insert_idx, :, insert_start:insert_start + insert_length]
                insert_msg = msg_tmp[insert_idx, :, insert_start:insert_start + insert_length]
                insert_mask = mask[insert_idx, :, insert_start:insert_start + insert_length]
            else:
                # If batch size is 1, use the same signal (for testing purposes)
                insert_segment = signal[..., insert_start:insert_start + insert_length]
                insert_watermark = watermark[..., insert_start:insert_start + insert_length]
                insert_msg = msg_tmp[..., insert_start:insert_start + insert_length]
                insert_mask = mask[..., insert_start:insert_start + insert_length]
                

            # Insert the segment into the signal
            signal = torch.cat([signal[...,:insert_pos],insert_segment,signal[..., insert_pos:]], dim=-1)

            watermark = torch.cat([watermark[...,:insert_pos],insert_watermark,watermark[..., insert_pos:]], dim=-1)

            if msg_tmp is not None:
                msg_tmp = torch.cat([msg_tmp[..., :insert_pos],insert_msg,msg_tmp[..., insert_pos:]], dim=-1)
            
            # Update the mask and watermark accordingly
            if torch.rand(1) > 0.5:
                mask = torch.cat([mask[..., :insert_pos],torch.zeros_like(mask[..., :insert_length]),mask[..., insert_pos:]], dim=-1)
            else: 
                mask = torch.cat([mask[..., :insert_pos],insert_mask,mask[..., insert_pos:]], dim=-1)


    # signal *= mask  # pad signal
    watermark *= mask  # Apply mask to the watermark

    if msg_tmp is not None:
        msg_tmp *= mask
    else:
        msg_tmp = torch.zeros(watermark.shape[0],0,watermark.shape[-1]).to(watermark.device)
    
    return signal, watermark, mask, msg_tmp
