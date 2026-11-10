import torch
import torch.nn.functional as F
import secrets

from .integrity_manifest import IntegrityManifest
from .keyed_schedule import ScheduleConfig, build_schedule


class MsgGenerator():

    def __init__(self, msg_cfg, master_key=None):

        self.msg_fix_length = msg_cfg.fix.length


        self.msg_temp_length = msg_cfg.temp.length
        self.msg_temp_interval = msg_cfg.temp.interval
        self.msg_temp_interval_range = msg_cfg.temp.interval_range


        self.msg_val_length = msg_cfg.val.length
        self.msg_val_interval = msg_cfg.val.interval
        self.msg_val_interval_range = msg_cfg.val.interval_range

        self.use_d_min_max = False
        self.schedule_version = "legacy_increment_v0"
        self.master_key = master_key or secrets.token_bytes(32)
        self._stream_counter = 0
        self.last_manifests = []


        if 'd_min' in msg_cfg.val:
            self.use_d_min_max = True
            self.d_min = msg_cfg.val.d_min
            self.d_max = msg_cfg.val.d_max
            self.val_increment = getattr(
                msg_cfg.val, "length_step", getattr(msg_cfg.val, "increment", 32)
            )



            self.schedule_version = getattr(
                msg_cfg.val, "schedule_version", "keyed_prf_v1"
            )
            if "schedule_version" not in msg_cfg.val and self.d_max == 2064:



                self.d_max = 1984


        self.msg_fix = None
        self.msg_temp = None
        self.msg_val = None
        self.msg_total = None

    def msg_generate(
        self,
        x,
        return_seg=False,
        stream_ids=None,
        creator_ids=None,
        start_segment_indices=None,
    ):
        b, _, t = x.shape
        if stream_ids is None:
            stream_ids = [f"stream-{self._stream_counter + i}" for i in range(b)]
        if creator_ids is None:
            creator_ids = ["training" for _ in range(b)]
        if start_segment_indices is None:
            start_segment_indices = [0 for _ in range(b)]
        if not (len(stream_ids) == len(creator_ids) == len(start_segment_indices) == b):
            raise ValueError("stream context lists must match batch size")
        self._stream_counter += b
        self.last_manifests = []


        self.msg_total = torch.zeros(b, 0, t).to(x.device)
        self.msg_seg = []


        if self.msg_fix_length > 0:

            self.msg_fix = self.msg_fix_random_generate(batch_size=b, msg_length=self.msg_fix_length)

            self.msg_fix = self.msg_fix.repeat(1, 1, t)
            self.msg_fix = self.msg_fix.to(x.device)
            self.msg_total = torch.cat((self.msg_total, self.msg_fix), dim=1)


        msg_temp_seg = []
        if self.msg_temp_length > 0:

            self.msg_temp, msg_temp_seg = self.random_rec_msg_generate(
                x,
                temp_msg_length=self.msg_temp_length,
                interval=self.msg_temp_interval,
                interval_range=self.msg_temp_interval_range
            )
            self.msg_temp = self.msg_temp.to(x.device)
            self.msg_total = torch.cat((self.msg_total, self.msg_temp), dim=1)
        self.msg_seg.append(msg_temp_seg)


        msg_val_seg = []
        if self.msg_val_length > 0:
            if self.use_d_min_max:


                self.msg_val, msg_val_seg = self.val_msg_generate(
                    x,
                    val_msg_bits=self.msg_val_length,
                    d_min=self.d_min,
                    d_max=self.d_max,
                    increment=self.val_increment,
                    stream_ids=stream_ids,
                    creator_ids=creator_ids,
                    start_segment_indices=start_segment_indices,
                )
            else:

                self.msg_val, msg_val_seg = self.circle_rec_msg_generate(
                    x,
                    temp_msg_length=self.msg_val_length,
                    interval=self.msg_val_interval,
                    interval_range=self.msg_val_interval_range
                )
            self.msg_val = self.msg_val.to(x.device)
            self.msg_total = torch.cat((self.msg_total, self.msg_val), dim=1)
        self.msg_seg.append(msg_val_seg)


        if return_seg:
            return self.msg_total, self.msg_fix, self.msg_temp, self.msg_val, self.msg_seg
        return self.msg_total, self.msg_fix, self.msg_temp, self.msg_val

    def msg_fix_random_generate(self, batch_size, msg_length):
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
                msg = torch.randint(0, 2, (temp_msg_length, 1), dtype=torch.float32)
                msg_seg.append([curr,curr+real_interval,msg.reshape(-1)])
                repeated_sequences[i, :, curr:curr+real_interval] += msg
                curr=curr+real_interval
        return repeated_sequences, msg_seg


    def val_msg_generate(
        self,
        x,
        val_msg_bits,
        d_min,
        d_max,
        increment=32,
        stream_ids=None,
        creator_ids=None,
        start_segment_indices=None,
    ):
        batch_size = x.shape[0]
        length = x.shape[-1]


        repeated_sequences = torch.zeros((batch_size, val_msg_bits, length))
        batch_segments = []
        stream_ids = stream_ids or [f"stream-{i}" for i in range(batch_size)]
        creator_ids = creator_ids or ["training" for _ in range(batch_size)]
        start_segment_indices = start_segment_indices or [0 for _ in range(batch_size)]


        for i in range(batch_size):
            schedule = None
            if self.schedule_version == "keyed_prf_v1":
                schedule = build_schedule(
                    self.master_key,
                    stream_ids[i],
                    creator_ids[i],
                    ScheduleConfig(d_min, d_max, increment),
                )
                initial_state = schedule.initial_state(val_msg_bits)
            else:
                initial_state = torch.randint(2 ** val_msg_bits, (1,)).item()

            curr_pos = 0
            curr_length = d_min
            segment_index = int(start_segment_indices[i])
            sample_segments = []
            while curr_pos < length:
                state_num = (initial_state + segment_index) % (2 ** val_msg_bits)
                seq = torch.tensor(
                    [int(bit) for bit in f"{state_num:0{val_msg_bits}b}"]
                )
                if schedule is not None:
                    curr_length = schedule.length_at(segment_index)
                else:
                    curr_length += increment
                    if curr_length >= d_max:
                        curr_length = curr_length - d_max + d_min
                emitted_length = min(curr_length, length - curr_pos)
                repeated_sequences[i, :, curr_pos:curr_pos + emitted_length] = seq.reshape(-1, 1)
                sample_segments.append([curr_pos, curr_pos + emitted_length, seq.reshape(-1)])
                curr_pos += emitted_length
                segment_index += 1

            batch_segments.append(sample_segments)
            self.last_manifests.append(IntegrityManifest(
                stream_id=str(stream_ids[i]),
                creator_id=str(creator_ids[i]),
                segment_count=segment_index,
                state_bits=val_msg_bits,
                initial_state=initial_state,
                schedule_version=self.schedule_version,
                d_min=d_min,
                d_max=d_max,
                length_step=increment,
            ))


        msg_segments = batch_segments[0] if batch_size == 1 else batch_segments
        return repeated_sequences, msg_segments


def post_process(x, window_size=5, threshold=0.1, min_duration=128, return_seg=False):

    x = F.sigmoid(x)

    B, num_states, T = x.shape
    output = torch.zeros_like(x, dtype=torch.long)
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.to(x.device)


    for state_idx in range(num_states):

        p = x[:, state_idx, :]


        p_smoothed = F.conv1d(p, kernel.view(1, 1, -1), padding=window_size // 2)





        p_corrected = torch.where(
            p_smoothed > 1 - threshold,
            torch.tensor(1.0, device=x.device),
            torch.where(
                p_smoothed < threshold,
                torch.tensor(0.0, device=x.device),
                p_smoothed
            )
        )


        output[:, state_idx, :] = (p_corrected > 0.5).long()

    if return_seg:

        segments = sequence_to_segments_fast(output)

        for batch_idx in range(B):
            batch_segment = segments[batch_idx]
            i = 0
            while i < len(batch_segment):
                curr_seg = batch_segment[i]
                duration = curr_seg[1] - curr_seg[0]

                if duration < min_duration:

                    prev_seg = batch_segment[i-1] if i > 0 else None
                    next_seg = batch_segment[i+1] if i < len(batch_segment)-1 else None


                    if prev_seg is not None and next_seg is not None:

                        if torch.equal(prev_seg[2],next_seg[2]):

                            new_seg = (prev_seg[0], next_seg[1], prev_seg[2])
                            batch_segment[i-1] = new_seg
                            batch_segment.pop(i)
                            batch_segment.pop(i)
                            i -= 1
                        else:

                            new_seg = (prev_seg[0], curr_seg[1], prev_seg[2])
                            batch_segment[i-1] = new_seg
                            batch_segment.pop(i)
                            i -= 1
                    elif prev_seg is not None:

                        new_seg = (prev_seg[0], curr_seg[1], prev_seg[2])
                        batch_segment[i-1] = new_seg
                        batch_segment.pop(i)
                        i -= 1
                    elif next_seg is not None:

                        new_seg = (curr_seg[0], next_seg[1], next_seg[2])
                        batch_segment[i] = new_seg
                        batch_segment.pop(i+1)
                i += 1
            segments[batch_idx] = batch_segment









        return output, segments
    else:
        return output
















































def sequence_to_segments_fast(output):
    B, C, T = output.shape
    segments = []


    output = output.permute(0, 2, 1)

    for b in range(B):

        state_changes = (output[b, 1:] != output[b, :-1]).any(dim=1)
        change_indices = torch.where(state_changes)[0]


        all_indices = torch.cat([
            torch.tensor([0], device=output.device),
            change_indices + 1,
            torch.tensor([T], device=output.device)
        ])


        batch_segments = []
        for i in range(len(all_indices) - 1):
            start = all_indices[i].item()

            end = all_indices[i + 1].item()
            state = output[b, start]
            batch_segments.append([start, end, state.detach()])

        segments.append(batch_segments)

    return segments


    return segments

def sequence_to_segments(output):
    B, C, T = output.shape
    segments = []

    for b in range(B):
        batch_segments = []

        current_state = output[b, :, 0]
        start = 0

        for t in range(1, T):
            state_t = output[b, :, t]

            if not (state_t == current_state).all():

                batch_segments.append([start, t-1, current_state.clone()])

                start = t
                current_state = state_t


        batch_segments.append([start, T-1, current_state.clone()])
        segments.append(batch_segments)

    return segments



def crop(signal, watermark, cfg, msg_tmp=None, clip=False, return_seg=False):
    assert (
        cfg.prob + cfg.shuffle_prob + cfg.pad_prob
        <= 1
    ), f"The sum of the probabilities {cfg.prob=} {cfg.shuffle_prob=} \
            {cfg.pad_prob=} should be less than 1"


    mask = torch.ones_like(watermark)
    p = torch.rand(1)
    if p < cfg.pad_prob:
        start = int(torch.rand(1) * 0.33 * watermark.size(-1))
        finish = int((0.66 + torch.rand(1) * 0.33) * watermark.size(-1))
        mask[:, :, :start] = 0
        mask[:, :, finish:] = 0
        if torch.rand(1) > 0.5:
            mask = 1 - mask
        signal *= mask




    elif (
        p < cfg.prob + cfg.pad_prob + cfg.shuffle_prob
    ):

        mask_size = round(watermark.shape[-1] * cfg.size)
        n_windows = int(
            torch.randint(1, cfg.max_n_windows + 1, (1,)).item()
        )
        window_size = int(mask_size / n_windows)
        for _ in range(n_windows):
            mask_start = torch.randint(0, watermark.shape[-1] - window_size, (1,))
            mask[:, :, mask_start: mask_start + window_size] = (
                0
            )

        if torch.rand(1) > 0.5:
            mask = 1 - mask

        if p < cfg.pad_prob + cfg.shuffle_prob:

            if signal.shape[0]>1:
                shuffle_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                while (shuffle_idx == torch.arange(signal.size(0))).any():
                    shuffle_idx = torch.randint(0, signal.size(0), (signal.size(0),))
            else:
                shuffle_idx=0



            signal = signal * mask + signal[shuffle_idx] * (1 - mask)





    elif p < cfg.prob + cfg.pad_prob + cfg.shuffle_prob + cfg.insert_prob:
        insert_num=int(torch.randint(1, cfg.max_n_windows, (1,)).item())
        for _ in range(insert_num):

            insert_len_min = int(cfg.insert_size[0])
            insert_len_max = int(cfg.insert_size[1])
            insert_start = int(0.11 * watermark.shape[-1])
            insert_end = int(0.88 * watermark.shape[-1])
            insert_length = torch.randint(insert_len_min, insert_len_max, (1,))
            insert_pos = torch.randint(insert_start, insert_end, (1,)).item()

            if insert_start + insert_length > signal.shape[-1]:
                insert_length=signal.shape[-1]-insert_start
            insert_length=insert_length-insert_length%100












            if signal.size(0) > 1:

                insert_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                while (insert_idx == torch.arange(signal.size(0))).any():
                    insert_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                insert_segment = signal[insert_idx, :, insert_start:insert_start + insert_length]
                insert_watermark = watermark[insert_idx, :, insert_start:insert_start + insert_length]
                insert_msg = msg_tmp[insert_idx, :, insert_start:insert_start + insert_length]
                insert_mask = mask[insert_idx, :, insert_start:insert_start + insert_length]
            else:

                insert_segment = signal[..., insert_start:insert_start + insert_length]
                insert_watermark = watermark[..., insert_start:insert_start + insert_length]
                insert_msg = msg_tmp[..., insert_start:insert_start + insert_length]
                insert_mask = mask[..., insert_start:insert_start + insert_length]



            signal = torch.cat([signal[...,:insert_pos],insert_segment,signal[..., insert_pos:]], dim=-1)

            watermark = torch.cat([watermark[...,:insert_pos],insert_watermark,watermark[..., insert_pos:]], dim=-1)

            if msg_tmp is not None:
                msg_tmp = torch.cat([msg_tmp[..., :insert_pos],insert_msg,msg_tmp[..., insert_pos:]], dim=-1)


            if torch.rand(1) > 0.5:
                mask = torch.cat([mask[..., :insert_pos],torch.zeros_like(mask[..., :insert_length]),mask[..., insert_pos:]], dim=-1)
            else:
                mask = torch.cat([mask[..., :insert_pos],insert_mask,mask[..., insert_pos:]], dim=-1)



    watermark *= mask

    if msg_tmp is not None:
        msg_tmp *= mask
    else:
        msg_tmp = torch.zeros(watermark.shape[0],0,watermark.shape[-1]).to(watermark.device)

    return signal, watermark, mask, msg_tmp
