import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from transformers import BertModel, BertTokenizer
from torchvision.models import googlenet

class GatedFusionNet(nn.Module):
    def __init__(self, config_path, num_classes):
        super().__init__()

        # 从配置文件加载参数
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        bert_path = self.config['bert_path']
        pretrained_model_path = self.config['pretrained_model_path']
        self.txt_dropout_p = self.config['training']['txt_dropout_p']
        
        # 使用GoogleNet替换ResNet
        base_model = googlenet(weights=None, aux_logits=False, init_weights=False)
        
        # 加载预训练权重，移除分类层
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        base_model.load_state_dict(state_dict, strict=False)

        # 提取特征部分并Flatten
        self.img_backbone = nn.Sequential(
            *list(base_model.children())[:-1],  # 去掉fc层
            nn.Flatten()
        )
        D = 1024  # GoogleNet最终特征维度调整为1024
            
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 修复模型加载方式
        self.txt_encoder = BertModel.from_pretrained(
            bert_path, 
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
        self.txt_proj = nn.Linear(self.txt_encoder.config.hidden_size, D)  # 投射为 512

        self.gate = nn.Sequential(
            nn.Linear(2 * D, D // 2),
            nn.ReLU(inplace=True),
            nn.Linear(D // 2, 1),
            nn.Sigmoid()
        )

        # 最终分类头
        self.classifier = nn.Linear(D, num_classes)  # 输入维度从1024改为2048

        # 新增相似度计算层
        #self.sim_net = nn.CosineSimilarity(dim=2)

        # 可学习的缩放因子和偏置
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, ocr_texts):
        v_img = self.img_backbone(images)  # [B,1024]
        
        # 文本特征提取
        enc = self.tokenizer(
        ocr_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
        )

        enc = {k: v.to(images.device) for k, v in enc.items()}

        v_txt = self.txt_encoder(**enc).pooler_output  
        v_txt = self.txt_proj(v_txt)  
        
        # 有30%的概率屏蔽文本特征，迫使模型更依赖图像特征进行分类。 
        # 这有助于防止模型过于依赖文本信息，可能导致过拟合。
        if self.training:
            mask = (torch.rand(v_txt.size(0), 1, device=images.device) > self.txt_dropout_p).float()
            v_txt = v_txt * mask
        
        cat = torch.cat([v_img, v_txt], dim=1)  # [B, 2D]
        g   = self.gate(cat)                    # [B, 1]
        fused   = g * v_img + (1 - g) * v_txt      # [B, D]
        
        # 修改融合方式
        #fused = torch.cat([v_img, v_txt], dim=1)  # [B, 2048]直接拼接

        ''' 整个batch计算的相似度矩阵,不知道为什么精度还提高了,明明引入了噪声，反而效果更好了
        # 计算相似度矩阵
        sim_matrix = self.sim_net(v_img.unsqueeze(1), v_txt.unsqueeze(0))  # [B,B]
        attn_weights = torch.softmax(sim_matrix, dim=1)  # 按行归一化
        # 相似度加权文本特征
        weighted_txt = torch.matmul(attn_weights, v_txt)  # [B,1024]
        '''

        '''        
        # 修改后的单样本相似度计算，有问题，还得考虑
        # 将图像和文本特征扩展维度为 [B,1,1024]
        img_exp = v_img.unsqueeze(1)  # [B,1,1024]
        txt_exp = v_txt.unsqueeze(1)  # [B,1,1024]
        
        # 计算单样本内相似度矩阵 [B,1,1]
        sim_matrix = self.sim_net(img_exp, txt_exp.permute(0,2,1))  # [B,1,1]
        attn_weights = torch.sigmoid(sim_matrix.squeeze())  # [B,1]
        
        # 加权当前样本的文本特征
        weighted_txt = attn_weights * v_txt  # [B,1024]
        '''

        '''
        #用相似度(标量权重[B,1])对某个图像对应的文本特征整体进行加权
        # 归一化特征
        norm_img = F.normalize(v_img, p=2, dim=1)
        norm_txt = F.normalize(v_txt, p=2, dim=1)
        
        # 计算余弦相似度
        sim_vector = (norm_img * norm_txt).sum(dim=1, keepdim=True)  # [B,1]
        attn_weights = torch.sigmoid(sim_vector)  # [B,1]

        # 加权文本特征
        weighted_txt = attn_weights * v_txt

        #拼接特征
        fused = torch.cat([v_img, weighted_txt], dim=1)  # [B,2048]
        '''
        return self.classifier(fused)


def build_model(config_path, num_classes):
    return GatedFusionNet(config_path=config_path, num_classes=num_classes)