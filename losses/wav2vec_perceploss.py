import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio





# 多层差异加权感知损失函数
class WeightedPerceptualLoss(torch.nn.Module):
    def __init__(self, selected_layers=[0,1,2, 3], loss_fn=torch.nn.MSELoss(), weights=[0.4,0.3,0.2,0.1], sample_rate=16000):
        """
        多层差异加权感知损失
        
        参数：
        - num_layers: Wav2Vec2 模型的 Transformer 层数
        - loss_fn: 用于比较特征差异的损失函数，默认使用 MSELoss（均方误差）
        - learnable_weights: 是否让层权重可学习，默认 False（固定权重）
        """
        super().__init__()  # 调用父类的初始化方法
        # 加载预训练模型和处理器
        local_model_path = "pretrained/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(local_model_path, local_files_only=True)
        self.model = Wav2Vec2Model.from_pretrained(local_model_path, local_files_only=True, output_hidden_states=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.sample_rate = sample_rate

        self.loss_fn = loss_fn
        self.selected_layers=selected_layers
        
        # 初始化各层的权重
        if weights:
            self.weights=torch.tensor(weights)
        else:
            self.weights=torch.ones(len(selected_layers))
        self.weights/=torch.sum(self.weights)  # 归一化权重

    def forward(self, waveform1, waveform2):
        """
        计算多层加权感知损失
        
        参数：
        - features1: 第一段音频的隐藏层特征 (list of Tensors)
        - features2: 第二段音频的隐藏层特征 (list of Tensors)
        
        返回：
        - 加权感知损失值 (torch.Tensor)
        """
        total_loss = 0.0

        B=waveform1.shape[0]
        
        # 预处理音频，转换为模型输入格式

        # 提取隐藏层特征
        outputs1 = self.model(waveform1)
        outputs2 = self.model(waveform2)
        
        features1 = outputs1.hidden_states
        features2 = outputs2.hidden_states

        # 遍历每一层的特征
        for i, layer_idx in enumerate(self.selected_layers):
            # 确保序列长度一致
            min_length = min(features1[i].shape[1], features2[i].shape[1])
            f1 = features1[layer_idx][:, :min_length, :]
            f2 = features2[layer_idx][:, :min_length, :]
            
            # 计算该层的损失并加权
            layer_loss = self.loss_fn(f1, f2)
            total_loss += self.weights[i] * layer_loss
        
        # 返回加权后的总损失
        return total_loss
    