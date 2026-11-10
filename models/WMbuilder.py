import julius
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .modules.seanet import SEANetWMEmbedder, SEANetEncoderKeepDimension
from .modules.demucs import Demucs, DemucsStreamer
from .modules.demucs_temp import DemucsTemp
from .modules.WM1d import Detector1d
from .modules.WM2d import Detector2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MsgProcessor(torch.nn.Module):

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:

        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)
        indices = indices.repeat(msg.shape[0], 1)
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)
        msg_aux = msg_aux.sum(dim=-2)
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )
        hidden = hidden + msg_aux
        return hidden

    def forward_without_repeat(self, msg: torch.Tensor) -> torch.Tensor:

        b,k,_=msg.shape
        msg=msg.reshape(b,k)
        indices = 2 * torch.arange(k).to(msg.device)
        indices.reshape(1, k)
        indices = indices.repeat(b, 1)
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)
        msg_aux = msg_aux.sum(dim=-2)
        return msg_aux.unsqueeze(-1)

    def forward_temp(self, msg: torch.Tensor) -> torch.Tensor:

        b,k,t = msg.shape
        indices = 2 * torch.arange(k).to(msg.device)
        indices = indices.reshape(1, k, 1)
        indices = indices.repeat(b, 1, t)
        indices = (indices + msg).long()
        indices = indices.permute(0,2,1).reshape(b*t,k)
        msg_aux = self.msg_processor(indices)
        msg_aux = msg_aux.sum(dim=-2)
        msg_aux = msg_aux.reshape(b,t,self.hidden_size).permute(0,2,1)
        return msg_aux




class WMEmbedder(nn.Module):
    def __init__(self, model_config, klass, *args, msg_fix_nbits: int = 0, msg_temp_nbits: int=0, msg_val_nbits: int=0, **kwargs):
        super(WMEmbedder, self).__init__()
        self.temp_msg_nbits = msg_temp_nbits
        if klass == 'demucs':
            self.model=Demucs(**model_config['demucs'])
        if klass == 'demucs_temp':
            self.model=DemucsTemp(**model_config['demucs_temp'])
        if klass == 'seanet':
            self.model=SEANetWMEmbedder(**model_config['seanet_embedder'])

        self.msg_processor=None
        self.temp_msg_processor = None
        self.temp_msg_processor=MsgProcessor(nbits=msg_fix_nbits+msg_temp_nbits+msg_val_nbits, hidden_size=model_config['temp_msg_dimension']['demucs'])
        self._message = None


    @property
    def message(self):
        return self._message

    def message(self, message):
        self._message = message

    def get_watermark(
        self,
        x,
        sample_rate = 16000,
        message= None,
        temp_message=None,
    ):
        length = x.size(-1)

        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)


        if self.temp_msg_processor is not None and temp_message is not None:
            msg_temp_hidden=self.temp_msg_processor.forward_temp(temp_message)
        else:
            msg_temp_hidden=None

        if self.msg_processor is not None and message is not None:
            msg_fix_hidden = self.msg_processor.forward_without_repeat(message)
        else:
            msg_fix_hidden=None

        watermark = self.model(x, msg_temp_hidden=msg_temp_hidden, msg_fix_hidden=msg_fix_hidden)




        return watermark[..., :length]


    def forward(self, x, sample_rate=None, message=None, alpha=1.0, temp_message=None):
        wm = self.get_watermark(x, message=message,temp_message=temp_message)

        return alpha*wm

class WMExtractor(torch.nn.Module):

    def __init__(self, model_config, klass, *args, nbits: int = 0, **kwargs):
        super().__init__()
        if klass=='seanet_detector':
            self.encoder = SEANetEncoderKeepDimension(**model_config['seanet_detector'])
            self.last_layer = torch.nn.Conv1d(self.encoder.output_dim, 1 + nbits, 1)
            self.detector = torch.nn.Sequential(self.encoder, self.last_layer)

        if klass=='wm1d':
            self.detector=Detector1d(msg_length=nbits+1, **model_config['WM1d'])
        if klass=='wm2d':
            self.detector=Detector2d(msg_length=nbits+1, **model_config['WM2d'])
        self.nbits = nbits

    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message_threshold: float = 0.5,
    ) -> Tuple[float, torch.Tensor]:
        det, rec = self.forward(x, sample_rate=sample_rate)

        detected_per_example = (det[:, 0, :] > 0.5).float().mean(dim=-1)

        detect_prob = detected_per_example.mean().cpu().item()
        message = torch.gt(rec, message_threshold).int()
        return detect_prob, message

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.detector(x)
        det = result[:, :1, :]
        rec = result[:, 1:, :]

        return det, rec
