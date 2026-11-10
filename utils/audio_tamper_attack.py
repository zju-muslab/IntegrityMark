import random
import torch

def _validate_audio(audio):
    if not isinstance(audio, torch.Tensor):
        raise ValueError("source_audio must be a torch.Tensor")
    if len(audio.shape) < 2:
        raise ValueError("source_audio must have at least two dimensions (channels, samples)")

def _random_range(audio_len, range_start, range_end):
    return random.randint(int(audio_len * range_start), int(audio_len * range_end))

def delete(source_audio, delete_start_range=[0.1, 0.4], delete_len_range=[16000, 24000]):
    _validate_audio(source_audio)
    audio_len = source_audio.shape[-1]
    pos_start = _random_range(audio_len, delete_start_range[0], delete_start_range[1])
    delete_len = random.randint(delete_len_range[0], delete_len_range[1])

    if pos_start + delete_len > audio_len-16000:
        delete_len=16000
        pos_start = random.randint(16000,audio_len - 16000-delete_len)

    pos_start=pos_start-pos_start%32
    delete_len=delete_len-delete_len%32

    audio_part1 = source_audio[..., :pos_start]
    audio_part2 = source_audio[..., pos_start + delete_len:]
    result = torch.cat([audio_part1, audio_part2], dim=-1)
    discrete_point = [pos_start]
    return result, discrete_point


def in_source_replace(source_audio, insertto_start_range=[0.2, 0.8], insertto_len_range=[0, 16000],
                      insertfrom_start_range=[0.2, 0.8], insertfrom_len_range=[16000, 32000]):

    _validate_audio(source_audio)
    audio_len = source_audio.shape[-1]

    loop_cnt=0
    while True:
        loop_cnt+=1
        assert loop_cnt<100
        insertto_start = _random_range(audio_len, insertto_start_range[0], insertto_start_range[1])
        insertto_len = random.randint(insertto_len_range[0], insertto_len_range[1])
        insertto_start=insertto_start-insertto_start%32
        insertto_len = insertto_len-insertto_len%32
        insertto_end=insertto_start+insertto_len
        if insertto_start + insertto_len > audio_len-8000:
            continue

        insertfrom_start = _random_range(audio_len, insertfrom_start_range[0], insertfrom_start_range[1])
        insertfrom_len = random.randint(insertfrom_len_range[0], insertfrom_len_range[1])

        insertfrom_start=insertfrom_start-insertfrom_start%32
        insertfrom_len = insertfrom_len-insertfrom_len%32
        insertfrom_end = insertfrom_start + insertfrom_len
        if insertfrom_start + insertfrom_len > audio_len-8000:
            continue

        if abs(insertto_start-insertfrom_start)>8000 and abs(insertto_end-insertfrom_end)>8000:
            break

    insertfrom = source_audio.clone()[..., insertfrom_start:insertfrom_start + insertfrom_len]
    part1 = source_audio[..., :insertto_start]
    part2 = source_audio[..., insertto_start + insertto_len:]
    result = torch.cat([part1, insertfrom, part2], dim=-1)
    discrete_points = [insertto_start, insertto_start + insertfrom_len]

    return result, discrete_points

def in_source_insert(source_audio, insertto_start_range=[0.2, 0.8],
                     insertfrom_start_range=[0.2, 0.8], insertfrom_len_range=[16000, 32000]):
    result, discrete_points = in_source_replace(source_audio,
                             insertto_start_range=insertto_start_range,
                             insertto_len_range=[0, 0],
                             insertfrom_start_range=insertfrom_start_range,
                             insertfrom_len_range=insertfrom_len_range)
    return result, discrete_points

def cross_source_replace(source_audio, tamper_audio, clip_position=[1/3, 1/2], replace_range=[0, 8000], clip_range=[16000,32000]):

    audio_len = source_audio.shape[-1]
    pos_start = int(audio_len * clip_position[0])
    pos_end = int(audio_len * clip_position[1])
    pos_start = pos_start - pos_start % 128
    pos_end = pos_end - pos_end % 128
    insert_pos = random.randint(pos_start, pos_end)
    replace_len = random.randint(replace_range[0], replace_range[1])
    clip_len = random.randint(clip_range[0], clip_range[1])

    replace_len=replace_len-replace_len%128
    clip_len=clip_len-clip_len%128


    if tamper_audio.shape[-1] < clip_len:

        repeat_times = (clip_len // tamper_audio.shape[-1]) + 1
        tamper_audio = tamper_audio.repeat(repeat_times)


    clip_start = random.randint(0, tamper_audio.shape[-1] - clip_len)
    insert_audio = tamper_audio[...,clip_start:clip_start+clip_len].reshape(1,1,-1)


    result = source_audio.clone()
    first_part = result[...,:insert_pos]
    second_part = result[...,insert_pos + replace_len:]
    result = torch.cat([first_part, insert_audio, second_part],dim=2)

    tamper_segs = [
            [0, insert_pos-1, 0.0],
            [insert_pos, insert_pos+clip_len-1, 1.0],
            [insert_pos+clip_len, result.shape[-1]-1,0.0]
        ]
    return result, tamper_segs

def cross_source_insert(source_audio, tamper_audio):
    result, tamper_segs = cross_source_replace(source_audio, tamper_audio, clip_position=[1/3,2/3], replace_range=[0,0])
    return result, tamper_segs

def cross_source_multi_insert(source_audio, tamper_audio, num_operations=2, clip_range=[16000, 24000]):
    result = source_audio.clone()
    tamper_segs = []


    audio_len = result.shape[0]



    insert_positions = []

    section_size = audio_len / num_operations
    for i in range(num_operations):

        section_center = (i + 0.5) * section_size
        insert_pos = section_center
        insert_positions.append(insert_pos)


    insert_positions.sort(reverse=True)
    current_segs = [[0,source_audio.shape[-1],0]]


    for pos in insert_positions:

        clip_len = random.randint(clip_range[0], clip_range[1])


        tamper_len = tamper_audio.shape[-1]
        if tamper_len > clip_len:
            start_pos = random.randint(0, tamper_len - clip_len)
            insert_audio = tamper_audio[..., start_pos:start_pos + clip_len]
        else:
            clip_len = tamper_len
            insert_audio = tamper_audio


        result, temp_segs = cross_source_replace(
            result,
            insert_audio,
            clip_position=[pos/audio_len, pos/audio_len],
            replace_range=[0, 0],
            clip_range=[clip_len, clip_len]
        )

        for idx, seg in enumerate(current_segs):
            if seg[0] >= pos:
                current_segs[idx][0] = seg[0] + clip_len
            if seg[1]>=pos:
                current_segs[idx][1] = seg[1]+clip_len



        tmp=current_segs[0][1]
        current_segs[0][1]=temp_segs[0][1]
        current_segs.append(temp_segs[1])
        current_segs.append([temp_segs[2][0],tmp,0])

        current_segs.sort(key=lambda x: x[0])












    return result, current_segs

def in_source_multi_replace(source_audio, num_operations=2,
                            insertto_start_range=[0.2, 0.8], insertto_len_range=[0, 16000],
                            insertfrom_start_range=[0.2, 0.8], insertfrom_len_range=[16000, 32000]):
    _validate_audio(source_audio)
    audio_len = source_audio.shape[-1]
    result = source_audio.clone()
    discrete_points = []


    section_size = audio_len / num_operations


    for i in range(num_operations):
        loop_cnt = 0
        while True:
            loop_cnt += 1
            if loop_cnt > 100:
                break


            section_start = i * section_size
            section_end = (i + 1) * section_size
            insertto_start = random.randint(int(section_start), int(section_end - 8000))
            insertto_len = random.randint(insertto_len_range[0], insertto_len_range[1])
            insertto_start = insertto_start - insertto_start % 32
            insertto_len = insertto_len - insertto_len % 32
            insertto_end = insertto_start + insertto_len

            if insertto_start + insertto_len > audio_len - 8000:
                continue


            insertfrom_start = _random_range(audio_len, insertfrom_start_range[0], insertfrom_start_range[1])
            insertfrom_len = random.randint(insertfrom_len_range[0], insertfrom_len_range[1])
            insertfrom_start = insertfrom_start - insertfrom_start % 32
            insertfrom_len = insertfrom_len - insertfrom_len % 32
            insertfrom_end = insertfrom_start + insertfrom_len

            if insertfrom_start + insertfrom_len > audio_len - 8000:
                continue


            if abs(insertto_start - insertfrom_start) > 8000 and abs(insertto_end - insertfrom_end) > 8000:
                break


        insertfrom = result.clone()[..., insertfrom_start:insertfrom_start + insertfrom_len]
        part1 = result[..., :insertto_start]
        part2 = result[..., insertto_start + insertto_len:]
        result = torch.cat([part1, insertfrom, part2], dim=-1)


        discrete_points.append(insertto_start)
        discrete_points.append(insertto_start + insertfrom_len)

    return result, discrete_points
