import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio






class WeightedPerceptualLoss(torch.nn.Module):
    def __init__(self, selected_layers=[0,1,2, 3], loss_fn=torch.nn.MSELoss(), weights=[0.4,0.3,0.2,0.1], sample_rate=16000):
        super().__init__()

        local_model_path = "pretrained/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(local_model_path, local_files_only=True)
        self.model = Wav2Vec2Model.from_pretrained(local_model_path, local_files_only=True, output_hidden_states=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.sample_rate = sample_rate

        self.loss_fn = loss_fn
        self.selected_layers=selected_layers


        if weights:
            self.weights=torch.tensor(weights)
        else:
            self.weights=torch.ones(len(selected_layers))
        self.weights/=torch.sum(self.weights)

    def forward(self, waveform1, waveform2):
        total_loss = 0.0

        B=waveform1.shape[0]




        outputs1 = self.model(waveform1)
        outputs2 = self.model(waveform2)

        features1 = outputs1.hidden_states
        features2 = outputs2.hidden_states


        for i, layer_idx in enumerate(self.selected_layers):

            min_length = min(features1[i].shape[1], features2[i].shape[1])
            f1 = features1[layer_idx][:, :min_length, :]
            f2 = features2[layer_idx][:, :min_length, :]


            layer_loss = self.loss_fn(f1, f2)
            total_loss += self.weights[i] * layer_loss


        return total_loss
