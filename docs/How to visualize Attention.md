# How to visualize Attention

Sources: 
- https://github.com/facebookresearch/dinov2
- https://github.com/facebookresearch/dino


在 dino 的第一个版本中，作者确实可视化了注意力图，并提供了代码。所以基本上，我们只需要对第二个版本的代码稍作修改，使其适应第一个版本。

目标是获取 Transformer（每个 head）在最后一层的注意力。然后，我们可以使用这些注意力得分来绘制热图。

在 Dinov2 仓库中，我们在 `dinov2/models/vision_transformer.py` 实现了 ViT。在这里，我们可以向 `DinoVisionTransformer` 类添加一个函数，用于提取最后的自注意力。因此，我们可以合并来自dinov2的 `forward_features` 方法和来自 dinov1 的 `get_last_selfattention` 方法。

```python
def get_last_self_attention(self, x, masks=None):
    # 判断输入x是否为列表类型，如果是，则处理列表中的每个元素
    if isinstance(x, list):
        return self.forward_features_list(x, masks)
        
    # 如果输入x不是列表，则准备带有掩码的token
    x = self.prepare_tokens_with_masks(x, masks)
    
    # 遍历模型中的每个block，只在最后一个block返回注意力
    for i, blk in enumerate(self.blocks):
        # 如果不是最后一个block，继续正常传递数据
        if i < len(self.blocks) - 1:
            x = blk(x)
        else: 
            # 如果是最后一个block，返回其注意力
            return blk(x, return_attention=True)
```

为了实现从Block类中获取注意力得分的功能，我们需要修改 `Block` 类的 `forward` 方法，以便它能接收一个 `return_attention` 参数。以下是如何进行这项修改的示例代码：

```python
    def forward(self, x: Tensor, return_attention=False) -> Tensor:
    # 定义一个内部函数处理注意力层的残差连接。使用LayerScale (ls1) 和注意力层 (attn) 处理归一化后的输入 (norm1(x))
    def attn_residual_func(x: Tensor) -> Tensor:
        return self.ls1(self.attn(self.norm1(x)))

    # 定义一个内部函数处理前馈网络层的残差连接。使用LayerScale (ls2) 和 MLP层 (mlp) 处理归一化后的输入 (norm2(x))
    def ffn_residual_func(x: Tensor) -> Tensor:
        return self.ls2(self.mlp(self.norm2(x)))
        
    # 如果return_attention为True，则返回处理过的注意力得分，这适用于获取模型中的注意力映射
    if return_attention:
        return self.attn(self.norm1(x), return_attn=True)
            
    # 如果模型处于训练阶段并且drop path概率大于0.1，使用随机深度降低网络的复杂度
    if self.training and self.sample_drop_ratio > 0.1:
        # 应用随机深度到注意力层的残差连接
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=attn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
        )
        # 应用随机深度到FFN层的残差连接
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=ffn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
        )
    elif self.training and self.sample_drop_ratio > 0.0:
        # 如果drop path概率小于0.1但大于0，应用drop_path方法到注意力和FFN层
        x = x + self.drop_path1(attn_residual_func(x))
        x = x + self.drop_path1(ffn_residual_func(x))  # 应该使用drop_path2，这里有待修正
    else:
        # 如果不在训练阶段或drop path概率为0，直接添加残差连接的结果
        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
    # 返回最终的输出
    return x
```

为了确保从`Block`派生的`NestedTensorBlock`也能正确处理返回注意力得分的功能，我们需要在`NestedTensorBlock`类的`forward`方法中实现相同的修改。

```python
def forward(self, x_or_x_list, return_attention=False):
    """
    处理输入张量或张量列表，可选择返回注意力得分。
    
    参数:
    x_or_x_list (Tensor 或 list[Tensor]): 输入数据可以是单个张量或张量列表。
    return_attention (bool): 如果为True，则返回注意力得分而不是正常的前向传播结果。
    """
    
    # 检查输入是否为单个张量
    if isinstance(x_or_x_list, Tensor):
        # 调用基类的forward方法，可选择传递return_attention参数以返回注意力得分
        return super().forward(x_or_x_list, return_attention)

    # 检查输入是否为张量列表
    elif isinstance(x_or_x_list, list):
        # 确保对嵌套张量处理的必要库xFormers已安装
        assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
        # 处理嵌套张量的特定方法调用
        return self.forward_nested(x_or_x_list)

    # 如果输入既不是张量也不是张量列表，则抛出异常
    else:
        raise AssertionError("Input must be a Tensor or a list of Tensors")
```

要完善整个体系结构以适应处理列表和使用XFORMERS的场景，我们需要在`NestedTensorBlock`类的`forward_nested`方法中实现类似的修改，以便它也能处理`return_attention`参数。此外，我们还需要更新`Attention`类的`forward`方法，以允许传递`return_attn`参数，以便在需要时返回注意力得分。

```python
def forward(self, x: Tensor, return_attn=False) -> Tensor:
    """
    处理输入张量，计算注意力得分并可选择返回它们。
    
    参数:
    x (Tensor): 输入特征数据，形状为(B, N, C)其中B是批量大小，N是序列长度，C是通道数。
    return_attn (bool): 如果为True，则返回注意力矩阵而不是正常的前向传播结果。
    """
    # 获取输入的维度
    B, N, C = x.shape
    # 计算查询、键、值（qkv）并调整其维度以适应多头注意力的需求
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    # 分离查询、键和值，对查询进行缩放
    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    # 计算注意力得分
    attn = q @ k.transpose(-2, -1)

    # 对注意力得分进行softmax操作并应用dropout
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # 计算输出特征
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    
    # 如果return_attn为True，则返回注意力得分
    if return_attn:
        return attn

    # 否则返回处理后的输出特征
    return x
```

Again, the `Attention` class is only the base class, so you have to also adjust the `MemEffAttention` class:

```python
class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        """
        一种内存效率高的注意力机制，可选地返回注意力矩阵。

        参数:
        x (Tensor): 输入张量，形状为(B, N, C)，其中B是批次大小，N是序列长度，C是通道数。
        attn_bias (Tensor, 可选): 用于注意力的可选偏置张量，通常用于相对位置嵌入或掩蔽。
        return_attn (bool): 如果为True，返回注意力矩阵而不是正常的前向传播输出。

        返回:
        Tensor: 在注意力计算之后的输出张量，或者如果return_attn为True，则是注意力矩阵。
        """
        # 检查是否安装了xFormers，以便使用像嵌套张量这样的高级功能
        if not XFORMERS_AVAILABLE:
            # 确保在xFormers不可用时，不使用attn_bias
            assert attn_bias is None, "使用嵌套张量需要安装xFormers"
            # 调用超类方法，并传递return_attn参数
            return super().forward(x, return_attn)

        # 分解输入尺寸
        B, N, C = x.shape
        # 计算查询、键和值张量
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # 将qkv张量分解为单独的查询、键和值张量
        q, k, v = unbind(qkv, 2)

        # 使用提供的或计算出的偏置应用内存效率高的注意力
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # 将输出投影回原始张量形状并应用dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

请注意，这种方法现在只有在不使用XFORMERS（嵌套张量）的情况下才有效。

最后，我们可以使用dinov1中的`visualize_attention.py`文件，并对其稍作修改来可视化热图！请记住，你需要将代码中的`img = Image.open('cow-on-the-beach2.jpg')`行更改为你的特定文件名，并且显然需要从dinov2的github仓库下载权重。

此外，注意这个奇怪的代码行`attentions[:, 10] = 0`。我发现对于一个特定的像素，在所有的注意力头上，注意力分数非常高。这就是为什么我打印最大的pixel_idx，并且为每张独立的图片手动*禁用*它。我真的不明白为什么会发生这种情况…

```python
# 版权声明和许可证信息
# 注意：原始代码位于 https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import os  # 导入操作系统接口模块
import sys  # 导入系统模块
import argparse  # 导入命令行解析模块
import cv2  # 导入OpenCV库
import random  # 导入随机数生成模块
import colorsys  # 导入颜色系统转换模块
import requests  # 导入HTTP库
from io import BytesIO  # 导入用于处理输入输出的模块

import skimage.io  # 导入用于图像操作的scikit-image库
from skimage.measure import find_contours  # 导入用于寻找图像轮廓的函数
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
from matplotlib.patches import Polygon  # 导入用于绘制多边形的类
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入处理图像的模块
from torchvision import transforms as pth_transforms  # 导入预处理模块
import numpy as np  # 导入NumPy库
from PIL import Image  # 导入PIL库处理图像
from dinov2.models.vision_transformer import vit_small, vit_large  # 导入Dinov2中的Vision Transformer模型

# 主程序入口
if __name__ == '__main__':
    image_size = (952, 952)  # 设置图像大小
    output_dir = '.'  # 设置输出目录
    patch_size = 14  # 设置patch的大小

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置设备为GPU或CPU

    # 初始化模型
    model = vit_large(
        patch_size=14,
        img_size=526,
        init_values=1.0,
        # ffn_layer="mlp",  # 可以选择前馈层的类型，这里被注释掉了
        block_chunks=0
    )

    # 加载模型权重
    model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))
    for p in model.parameters():
        p.requires_grad = False  # 冻结模型参数，不进行梯度更新
    model.to(device)  # 将模型移动到指定的设备
    model.eval()  # 设置模型为评估模式

    # 加载并处理图像
    img = Image.open('cow-on-the-beach2.jpg')  # 打开图像文件
    img = img.convert('RGB')  # 转换图像为RGB格式
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),  # 重设图像大小
        pth_transforms.ToTensor(),  # 将图像转换为Tensor
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 归一化处理
    ])
    img = transform(img)  # 应用变换
    print(img.shape)

    # 使图像尺寸适配patch大小
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加一个批次维度

    # 计算特征图的宽度和高度
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)

    # 获取模型最后一层的注意力分数
    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # 获取头部数量
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # 重塑注意力分数
    print(torch.max(attentions, dim=1))  # 打印最大注意力值
    attentions[:, 283] = 0  # 将特定像素的注意力值设为0

    attentions = attentions.reshape(nh, w_featmap, h_featmap)  # 重塑注意力图
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()  # 上采样注意力图并转为numpy数组

    # 保存注意力热图
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")  # 设置文件名
        plt.imsave(fname=fname, arr=attentions[j], format='png')  # 保存热图
        print(f"{fname} saved.")  # 打印保存信息
```





