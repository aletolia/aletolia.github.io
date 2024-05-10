# Attention Generation Work Flow

这个文档描述了如何从网络中计算注意力分数，可以是 backbone，也可以是从 neck 中

- 首先仍然以 CLAM 为例子

注意力网络的定义

```python
"""
注意力网络（无门控，包含两个全连接层）
参数：
    L: 输入特征维度
    D: 隐藏层维度
    dropout: 是否使用dropout（概率p = 0.25）
    n_classes: 类别数
"""
class Attn_Net(nn.Module):
    # 构造函数，定义网络结构。
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()  # 调用父类的构造函数，初始化模块。
        self.module = [
            nn.Linear(L, D),  # 定义第一个全连接层，从输入特征维度L到隐藏层维度D。
            nn.Tanh()]  # 使用Tanh激活函数。

        if dropout:
            self.module.append(nn.Dropout(0.25))  # 如果启用dropout，添加一个dropout层，丢弃率为0.25。

        self.module.append(nn.Linear(D, n_classes))  # 添加第二个全连接层，从隐藏层维度D到输出类别数n_classes。
      
        self.module = nn.Sequential(*self.module)  # 将定义的层组合成一个顺序模型。
  
    def forward(self, x):
        return self.module(x), x  # 前向传播函数，返回注意力网络的输出和原始输入x。返回形式为N x n_classes。
```

应该说，这个网络的定义并非严格意义上的 `attention_net` ，接下来则是“门控注意力”

```python
class Attn_Net_Gated(nn.Module):
    # 构造函数，定义网络结构。
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()  # 调用父类的构造函数，初始化模块。
        # 定义第一组注意力机制的层，使用 Tanh 激活函数。
        self.attention_a = [
            nn.Linear(L, D),  # 定义全连接层，从输入特征维度L到隐藏层维度D。
            nn.Tanh()]  # 应用 Tanh 激活函数。

        # 定义第二组注意力机制的层，使用 Sigmoid 激活函数。
        self.attention_b = [nn.Linear(L, D),  # 定义全连接层。
                            nn.Sigmoid()]  # 应用 Sigmoid 激活函数来实现门控。

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))  # 如果启用 dropout，为第一组添加 dropout 层。
            self.attention_b.append(nn.Dropout(0.25))  # 为第二组添加 dropout 层。

        self.attention_a = nn.Sequential(*self.attention_a)  # 将第一组层组合成顺序模型。
        self.attention_b = nn.Sequential(*self.attention_b)  # 将第二组层组合成顺序模型。
      
        self.attention_c = nn.Linear(D, n_classes)  # 定义第三个全连接层，用于将经过门控的特征映射到类别数。

    def forward(self, x):
        a = self.attention_a(x)  # 通过第一组层处理输入数据 x。
        b = self.attention_b(x)  # 通过第二组层处理输入数据 x。
        A = a.mul(b)  # 将两组输出的结果相乘，实现门控机制。
        A = self.attention_c(A)  # 将门控后的特征通过第三层映射到类别输出。
        return A, x  # 返回类别输出和原始输入。
```

这里的 `Attn_Net_Gate` 实际上实现了下面的式子

$$
\begin{gathered}
a_i, k=\frac{\exp \left\{W_{\mathrm{a}, i}\left(\tanh \left(V_{\mathrm{a}} \mathbf{h}_k\right) \odot \operatorname{sigm}\left(U_{\mathrm{a}} \mathbf{h}_k\right)\right)\right\}}{\sum_{j=1}^K \exp \left\{W_{\mathrm{a}, i}\left(\tanh \left(V_{\mathrm{a}} \mathbf{h}_j\right) \odot \operatorname{sigm}\left(U_{\mathrm{a}} \mathbf{h}_j\right)\right)\right\}} \\
\mathbf{h}_{\mathrm{slide}, i}=\sum_{k=1}^K a_{i, k} \mathbf{h}_k
\end{gathered}
$$

如原文所述：注意网络由若干堆叠的完全连接层组成；如果我们将注意网络的前两层 $U_{\mathrm{a}} \in \mathbb{R}^{256 \times 512}$ 和 $V_{\mathrm{a}} \in \mathbb{R}^{256 \times 512}$ 以及 $W_1$ 一起视为所有类共享的注意力骨干的一部分，那么注意网络将分为 $N$ 个并行的注意力分支 $W_{\mathrm{a}, 1}, \ldots, W_{\mathrm{a}, \mathrm{N}} \in \mathbb{R}^{1 \times 256}$。

## CLAM_SB 架构的定义

在构造上，`CLAM_SB` 在定义时定义了下面的几个 layer

```python
class CLAM_SB(nn.Module):
    # 构造函数，初始化网络结构。
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()  # 调用父类的构造函数。
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]  # 根据网络尺寸配置获取维度参数。
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]  # 定义全连接层序列。
      
        # 根据是否使用门控选择相应的注意力网络。
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)  # 创建注意力网络。
        self.classifiers = nn.Linear(size[1], n_classes)  # 定义主分类器。
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]  # 为每个类别定义实例分类器。
        self.instance_classifiers = nn.ModuleList(instance_classifiers)  # 创建实例分类器列表。
        self.k_sample = k_sample  # 设置采样数量。
        self.instance_loss_fn = instance_loss_fn  # 设置实例级损失函数。
        self.n_classes = n_classes  # 设置类别数量。
        self.subtyping = subtyping  # 设置是否为亚型问题。
```

- `attention_net`：由门控注意力层和一层全连接层构成
- `classifiers`：一个分类头

前向传播中，即通过了一次 `attention_net`，返回注意力分数 A 后对 A 进行 `softmax`

```python
def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # 通过注意力网络处理输入h，得到注意力得分A和更新后的特征h，NxK。
        A = torch.transpose(A, 1, 0)  # 转置A以匹配后续操作的维度要求，KxN。
      
        if attention_only:
            return A  # 如果仅需要注意力得分，则直接返回A。
  
        A_raw = A  # 保存原始的注意力得分。
        A = F.softmax(A, dim=1)  # 对N维进行softmax操作，归一化注意力得分
```

进行迭代，迭代过程中进行实例评估，调用

```python
instance_loss, preds, targets = self.inst_eval(A, h, classifier)
```

首先检查输入维度

```python
if len(A.shape) == 1:
            A = A.view(1, -1)  # 调整A的形状。
```

检查输入张量 `A` 的形状。如果 `A` 只有一个维度，则将其重新整形为具有两个维度的张量。

```python
top_p_ids = torch.topk(A, self.k_sample)[1][-1]  # 获取正样本的top k样本索引。
top_p = torch.index_select(h, dim=0, index=top_p_ids)  # 根据索引选择正样本。
top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]  # 获取负样本的top k样本索引。
top_n = torch.index_select(h, dim=0, index=top_n_ids)  # 根据索引选择负样本。
```

`top_p_ids = torch.topk(A, self.k_sample)[1][-1]`: 使用 `torch.topk` 函数找到 `A` 中值最大的 `k_sample` 个元素的索引并存储在 `top_p_ids` 中。这些索引对应于最有可能属于正样本的数据点。

`top_p = torch.index_select(h, dim=0, index=top_p_ids)`: 使用 `torch.index_select` 函数根据 `top_p_ids` 中的索引从嵌入张量 `h` 中选取对应的行，得到 `k_sample` 个最有可能属于正样本的嵌入，并存储在 `top_p` 中。

`top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]`: 类似于选择正样本，这行代码找到 `A` 中值最小的 `k_sample` 个元素的索引并存储在 `top_n_ids` 中，这些索引对应于最有可能属于负样本的数据点。

`top_n = torch.index_select(h, dim=0, index=top_n_ids)`: 类似于选择正样本的嵌入，根据 `top_n_ids` 中的索引从嵌入张量 `h` 中选取对应的行，得到 `k_sample` 个最有可能属于负样本的嵌入，并存储在 `top_n` 中。

- 根据之前的代码，`x` 是原始输入

接下来创建用于后续计算损失的正负样本目标：

```python
p_targets = self.create_positive_targets(self.k_sample, device)
n_targets = self.create_negative_targets(self.k_sample, device)
```

这分别是两个静态方法

```python
@staticmethod
    def create_positive_targets(length, device):
        # 创建正样本目标。
        return torch.full((length,), 1, device=device).long()
```

该静态方法用于创建正样本目标。它接收两个参数：

- `length`: 要创建的正样本目标的数量 (与 `k_sample` 相同)。
- `device`: 张量所在的设备信息 (CPU 或 GPU)。

函数内部使用 `torch.full` 函数创建一个指定形状 (`(length,)`) 的全 1 张量。该张量中的所有元素都设置为 1，表示正样本的目标值。 `device=device` 参数确保创建的张量位于与输入数据相同的设备上

```python
all_targets = torch.cat([p_targets, n_targets], dim=0)
all_instances = torch.cat([top_p, top_n], dim=0)
```

- `all_targets = torch.cat([p_targets, n_targets], dim=0)`: 使用 `torch.cat` 函数将正负样本目标张量 (`p_targets` 和 `n_targets`) 在第 0 维 (batch 维度) 进行拼接，得到包含所有样本目标的张量 `all_targets`。
- `all_instances = torch.cat([top_p, top_n], dim=0)`: 类似于合并目标，将正负样本的嵌入张量 (`top_p` 和 `top_n`) 在第 0 维进行拼接，得到包含所有样本实例的张量 `all_instances`。

```python
ogits = classifier(all_instances)
all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
```

- `logits = classifier(all_instances)`: 将所有样本实例 (`all_instances`) 送入分类器 (`classifier`) 进行分类，得到 logits (未归一化的类别概率)
- `torch.topk(logits, 1, dim=1)`: 该部分使用 `torch.topk` 函数找到 logits 中每个样本得分最高的元素 (top 1) 的索引，并将其存储在张量中
- `[1]`: 取 `torch.topk` 函数返回结果的第二个元素，即包含每个样本得分最高元素的索引张量。
- `.squeeze(1)`: 去除索引张量的第二维 (因为我们只关心每个样本的单一预测类别)。 这将得到一个包含所有样本预测类别的张量 `all_preds`

计算损失函数（交叉熵）

```python
instance_loss = self.instance_loss_fn(logits, all_targets)
```

## Transformer-Explainability

以 DeiT 的可视化为例，首先使用 `transforms` 进行图像尺寸的转化

```python
# 设置图像预处理的归一化操作，指定均值和标准差
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# 定义一个图像转换流程，包括缩放到224x224、转换为张量、归一化处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 将图像转换为Tensor
    normalize,                      # 应用归一化
])
```

预设了一个热图生成函数，输入参数是 `img` 和 `mask` ，首先将 `mask` 转换为 heatmap

```python
def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # 创建热图
```

这里指将 `mask` 投影到 256 位色彩上，颜色映射使用 `COLORMAP_JET`

然后归一化转换为浮点数后叠加到源图像上

```python
heatmap = np.float32(heatmap) / 255  # 将热图转换为0到1的浮点数
cam = heatmap + np.float32(img)  # 将热图叠加到原始图像上
cam = cam / np.max(cam)  # 归一化处理以增强显示效果
return cam  # 返回叠加后的图像
```

因此实际上模型的难点在于

1. 取出每一层的注意力数值
2. 按照加权（深度泰勒分解）方式解释注意力在不同 patch 处的贡献程度

初始化模型，设置为 `eval()` 模式

```python
model = vit_LRP(pretrained=True).cuda()
model.eval()
attribution_generator = LRP(model)
```

乘此机会，也来看一下 ViT 的一般架构，下面是 `vit_LRP`  的模型定义，首先是输出联合注意力的函数

```python
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # 从第一个矩阵的维度中计算出token的数量和批次大小。
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]

    # 创建一个单位矩阵，并扩展它以匹配批次大小，然后将其移动到相应的设备上。
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)

    # 为每层的注意力矩阵添加一个单位矩阵，以包括残差连接。
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]

    # 初始化联合注意力矩阵为指定起始层的注意力矩阵。
    joint_attention = all_layer_matrices[start_layer]

    # 从起始层开始，逐层计算联合注意力矩阵，通过矩阵乘法将当前层的注意力矩阵与之前累积的联合注意力矩阵相乘。
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)

    # 返回计算得到的最终联合注意力矩阵。
    return joint_attention
```

1. **初始化变量**：首先从输入的多层注意力矩阵列表（`all_layer_matrices`）的第一项中获取 token 的数量和批次大小。
2. 接着创建一个单位矩阵，并将其扩展到与输入矩阵相同的批次大小，并确保其在相同的计算设备上。
3. **添加残差连接**：为了考虑残差连接，函数将每一层的注意力矩阵与单位矩阵相加。这样做可以帮助改善训练过程中的梯度流动和模型的整体性能。
4. **计算联合注意力**：从指定的起始层开始，通过矩阵连乘操作累积计算联合注意力。**这个过程中，每一层的注意力矩阵会与前一层的联合注意力矩阵相乘，逐步构建从起始层到当前层的全局注意力映射。**
5. **输出最终结果**：最终，函数输出从指定起始层到最后一层的联合注意力矩阵，这个矩阵包含了经过所有这些层的信息流的累积影响。

接着定义了一个类，是一个多层感知机，主要用于转换特征维度

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        # 如果没有指定输出特征数，就使用输入特征数
        out_features = out_features or in_features
        # 如果没有指定隐藏层特征数，也使用输入特征数
        hidden_features = hidden_features or in_features
        # 定义第一个全连接层
        self.fc1 = Linear(in_features, hidden_features)
        # 定义激活函数，这里使用GELU激活函数
        self.act = GELU()
        # 定义第二个全连接层
        self.fc2 = Linear(hidden_features, out_features)
        # 定义dropout层，用于减少过拟合
        self.drop = Dropout(drop)

    def forward(self, x):
        # 输入通过第一个全连接层
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # 应用dropout
        x = self.drop(x)
        # 输入通过第二个全连接层
        x = self.fc2(x)
        # 再次应用dropout
        x = self.drop(x)
        # 返回最终结果
        return x

    def relprop(self, cam, **kwargs):
        # 反向传播分析的最后一步，逆过程应用dropout
        cam = self.drop.relprop(cam, **kwargs)
        # 逆过程通过第二个全连接层
        cam = self.fc2.relprop(cam, **kwargs)
        # 逆过程应用激活函数
        cam = self.act.relprop(cam, **kwargs)
        # 逆过程通过第一个全连接层
        cam = self.fc1.relprop(cam, **kwargs)
        # 返回最终的相关性传播结果
        return cam
```

值得一提的是这里使用的 `relprop` 函数，用于模型解释性分析，主要是反向追踪输入特征对输出的影响，过程与前向传播相反

接下来是注意力块的定义：

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # 头的数量
        head_dim = dim // num_heads  # 计算每个头的维度
        self.scale = head_dim ** -0.5  # 缩放因子，用于稳定训练

        # 定义用于矩阵运算的函数
        self.matmul1 = einsum('bhid,bhjd->bhij')  # Q*K^T
        self.matmul2 = einsum('bhij,bhjd->bhid')  # A*V

        # 定义全连接层，用于生成查询Q、键K和值V
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)  # 注意力dropout
        self.proj = Linear(dim, dim)  # 最后的投影层
        self.proj_drop = Dropout(proj_drop)  # 投影层的dropout
        self.softmax = Softmax(dim=-1)  # softmax用于计算注意力权重

        # 用于保存注意力权重和中间梯度的属性
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    # 以下方法用于存取注意力权重、值和梯度等中间结果
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        # 对输入数据重新排列和分配头
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        # 计算注意力分数并应用softmax和dropout
        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        # 根据注意力权重和值向量计算输出
        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        # 反向传播方法，用于相关性传播分析，即解释模型决策过程
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)
```

类似 ViT 的原始定义，要如何实现将输入转换为 `Q,K,V` 三个矩阵？最简单的实现方法就是通过线性投影将输入投影到维度\*3 的维度，然后再分割开

```python
self.qkv = Linear(dim, dim * 3, bias=qkv_bias)

qkv = self.qkv(x)
q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
```

保存 `v`

```python
self.save_v(v)

def save_v(self, v):
	self.v = v
```

计算注意力分数并保存

```python
dots = self.matmul1([q, k]) * self.scale
attn = self.softmax(dots)
attn = self.attn_drop(attn)

self.save_attn(attn)

def save_attn(self, attn):
    self.attn = attn
```

下面是重头戏，在网络中 register 了一个 hook

```python
attn.register_hook(self.save_attn_gradients)

def save_attn_gradients(self, attn_gradients):
    self.attn_gradients = attn_gradients
```

在 PyTorch 中，`register_hook` 方法用于在一个 `torch.Tensor` 上 register 一个反向传播钩子。这个钩子是一个函数，它会在反向传播期间当该张量的梯度被计算时被调用，用于在反向传播时捕获并保存注意力权重 `attn` 的**梯度**

当 pytorch 在计算这个变量的梯度时，这个 hook 会被自动调用，然后把梯度传到这个函数中

接着将这些注意力分数（权重）应用的 `v` （Value）上（进行点积）

```python
out = self.matmul2([attn, v])
```

使用 `rearrange` 函数来改变 `out` 张量的形状。这个操作的目的是将不同头的输出合并为单个维度，从而为后续的全连接层做准备

```python
out = rearrange(out, 'b h n d -> b n (h d)')
```

全连接层输出提取好的特征

```python
out = self.proj(out)
out = self.proj_drop(out)
```

添加 `Dropout` 后的 `out` 可以在训练过程中提高模型的泛化能力，避免在训练数据上过度拟合

同时在这个注意力块中，也定义了一个相关性传播方法：

```python
def relprop(self, cam, **kwargs):
    # 对输出层的Dropout进行反向传播
    cam = self.proj_drop.relprop(cam, **kwargs)
    # 对输出层的全连接层进行反向传播
    cam = self.proj.relprop(cam, **kwargs)
    # 将cam张量的格式从扁平化特征重新排列为多头格式，以匹配多头注意力的原始维度
    cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

    # 反向传播通过注意力机制的第二部分，即A*V，其中A是注意力权重，V是值
    (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
    # 由于相关性传播需要考虑到每个输入的贡献，这里将cam分成两半，确保总相关性保持不变
    cam1 /= 2
    cam_v /= 2

    # 保存反向传播的值向量V的相关性
    self.save_v_cam(cam_v)
    # 保存反向传播的注意力权重A的相关性
    self.save_attn_cam(cam1)

    # 对注意力权重应用的Dropout进行反向传播
    cam1 = self.attn_drop.relprop(cam1, **kwargs)
    # 使用softmax的反向传播方法进一步传播相关性
    cam1 = self.softmax.relprop(cam1, **kwargs)

    # 反向传播通过注意力机制的第一部分，即Q*K^T，其中Q是查询，K是键
    (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
    cam_q /= 2
    cam_k /= 2

    # 将Q、K、V的相关性张量重新组合为一个张量，准备传递给QKV的全连接层的反向传播
    cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

    # 对QKV全连接层进行相关性反向传播
    return self.qkv.relprop(cam_qkv, **kwargs)
```

然后组装成 Block

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        # 第一个层归一化
        self.norm1 = LayerNorm(dim, eps=1e-6)
        # 注意力机制模块，包含前面定义的 Attention 类
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # 第二个层归一化
        self.norm2 = LayerNorm(dim, eps=1e-6)
        # MLP模块，包含前面定义的 Mlp 类
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # 加法操作，用于实现残差连接
        self.add1 = Add()
        self.add2 = Add()
        # 克隆操作，用于复制输入
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        # 克隆输入，准备进行第一次残差连接
        x1, x2 = self.clone1(x, 2)
        # 第一次残差连接，注意力机制后的输出与原始输入相加
        x = self.add1([x1, self.attn(self.norm1(x2))])
        # 克隆结果，准备进行第二次残差连接
        x1, x2 = self.clone2(x, 2)
        # 第二次残差连接，MLP后的输出与上一步的输出相加
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        # 返回最终结果
        return x

    def relprop(self, cam, **kwargs):
        # 反向传播相关性传播，从第二个残差连接开始
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        # 反向传播相关性传播，处理第一个残差连接
        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        # 返回最终的相关性传播结果
        return cam

```

- 初始化时，定义了两个层归一化层，一个注意力模块，和一个MLP模块。每个模块之间使用残差连接，通过加法操作来融合模块输出和输入
- 输入数据首先被克隆以用于残差连接。数据流经第一个层归一化和注意力模块，然后与原始输入相加。接着，结果再次经过层归一化和 MLP 处理，最后与之前的输出相加，形成最终输出

然后是图像 embedding 模块

```python
class PatchEmbed(nn.Module):
    """ 图像到块嵌入
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 确保图像尺寸和块尺寸是二元组形式
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 计算整个图像中的块数量
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 定义一个卷积层，用于从每个图像块中提取嵌入
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查输入图像尺寸是否与模型预期匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 应用卷积层，然后重新排列输出以形成嵌入向量
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        # 逆转换操作，用于相关性传播分析
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        # 将相关性传播应用到卷积层
        return self.proj.relprop(cam, **kwargs)
```

使用卷积层将每个块转换成嵌入向量，然后调整向量的排列顺序以适合后续的处理

接下来是一个完整的 ViT 实现

```python
class VisionTransformer(nn.Module):
    """ 视觉 Transformer，支持补丁或混合 CNN 输入阶段
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes  # 类别数
        self.num_features = self.embed_dim = embed_dim  # 特征数，与嵌入维度相同，用于与其他模型保持一致
      
        # 初始化补丁嵌入模块
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 初始化位置嵌入和类标记
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 初始化多个 Transformer 编码块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        # 最后的层归一化
        self.norm = LayerNorm(embed_dim)
        # 根据参数选择使用 MLP 或单个线性层作为分类头
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = Linear(embed_dim, num_classes)

        # 初始化权重
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # 辅助操作
        self.pool = IndexSelect()
        self.add = Add()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # 扩展类标记和合并类标记和补丁嵌入
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        # 应用 Transformer 编码块
        for blk in self.blocks:
            x = blk(x)

        # 应用层归一化和池化操作，然后通过分类头输出最终结果
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x
```

```python
def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
    # 基础的相关性传播，从分类头开始反向传播
    cam = self.head.relprop(cam, **kwargs)
    cam = cam.unsqueeze(1)
    cam = self.pool.relprop(cam, **kwargs)
    cam = self.norm.relprop(cam, **kwargs)

    # 反向通过所有 Transformer 编码块
    for blk in reversed(self.blocks):
        cam = blk.relprop(cam, **kwargs)

    # 根据不同的方法参数，执行不同的相关性传播分析策略
    if method == "full":
        (cam, _) = self.add.relprop(cam, **kwargs)
        cam = cam[:, 1:]
        cam = self.patch_embed.relprop(cam, **kwargs)
        cam = cam.sum(dim=1)  # 在通道维度上求和
        return cam

    elif method == "rollout":
        # 计算所有编码块的平均注意力权重，用于产生整体的注意力“rollout”图
        attn_cams = []
        for blk in self.blocks:
            attn_heads = blk.attn.get_attn_cam().clamp(min=0)
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            attn_cams.append(avg_heads)
        cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
        cam = cam[:, 0, 1:]
        return cam

    elif method == "transformer_attribution" or method == "grad":
        # 使用 Transformer 编码块的梯度和注意力权重计算最终的贡献图
        cams = []
        for blk in self.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        cam = rollout[:, 0, 1:]
        return cam

    # 分析特定层的注意力权重，可以选定最后一层或第二层
    elif method == "last_layer":
        cam = self.blocks[-1].attn.get_attn_cam()
        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        if is_ablation:
            grad = self.blocks[-1].attn.get_attn_gradients()
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        cam = cam[0, 1:]
        return cam
```

书接上文，在定义了 `vit_LRP` 之后，定义了函数 `generate_visualization`

```python
def generate_visualization(original_image, class_index=None):
    # 生成指定类索引的LRP属性图
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
```

```python
class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # 将模型设置为评估模式

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)  # 获得模型的输出
        kwargs = {"alpha": 1}  # 设置LRP方法的额外参数

        # 如果没有指定index，则自动选择输出概率最大的类别
        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # 创建一个one-hot编码的向量，用于指定感兴趣的输出类别
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)  # 乘以模型的输出并求和，为反向传播准备

        # 清除之前的梯度并进行反向传播
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)
```

主要是这几点

- **One-hot 编码**：创建一个 one-hot 编码的向量，标记目标类别的位置。
- **梯度计算**：将 one-hot 编码的结果与模型输出相乘并求和，以便在这个点上计算梯度。
- **反向传播**：对得到的结果执行反向传播，计算每个输入对输出类别的影响

这样，我们就获得了最终的注意力分数，然后是对注意力分数以及图像的处理

```python
# 将属性图的形状调整为 1x1x14x14，准备进行上采样
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    # 使用双线性插值将属性图的尺寸上采样到原始图像的尺寸，这里假设原始图像尺寸为224x224
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    # 调整属性图的形状到224x224，并将其移动到CPU内存中，转换为NumPy数组
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    # 对属性图进行归一化处理，将其值归一化到0和1之间，以便更好地可视化
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    # 将原始图像的维度从CxHxW转换为HxWxC，然后转换为NumPy数组
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    # 对原始图像也进行归一化处理，将其值归一化到0和1之间
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())

    # 使用热图叠加函数将属性图覆盖在原始图像上，生成最终的可视化效果
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    # 将最终的可视化结果转换为0-255的RGB格式，适合保存或显示
    vis = np.uint8(255 * vis)
    # 将图像从RGB格式转换为OpenCV使用的BGR格式
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    # 返回最终的可视化图像
    return vis
```
这个方法的在整体的构建上是和 ViT 的模型构建一起建立的，众所周知，在建立 ViT 模型时，一般会分成下面几个模块来分别定义

1. 分类投影器

```python
class Mlp(nn.Module)
```
2. 注意力模块

```python
class Attention(nn.Module)
```
3. 组装模块

```python
class Block(nn.Module)
```
4. 图像编码模块

```python
class PatchEmbed(nn.Module)
```

最后整合成完整的 ViT

```python
class VisionTransforme(nn.Module)
```

因此在进行相关性传播时也是如此，以 `Attention` 模块为例，首先定义了在前向传播和反向传播中的 hook

```python
def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R
```

接着在每个模块中都重载这个函数，使得相关性传播同样会随着网络深度的加深而加深

```python
def relprop(self, cam, **kwargs):
    # 通过投影下降层传播相关性
    cam = self.proj_drop.relprop(cam, **kwargs)
    # 通过投影层传播相关性
    cam = self.proj.relprop(cam, **kwargs)
    # 重新排列 cam 张量，从 (batch, num_patches, num_heads*depth) 格式变为 (batch, heads, num_patches, depth)
    cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)
    # attn = A*V，计算注意力得分与值的相关性
    (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
    cam1 /= 2  # 将 cam1 的相关性除以 2
    cam_v /= 2  # 将 cam_v 的相关性除以 2
    # 存储 V 和注意力得分的相关性图
    self.save_v_cam(cam_v)
    self.save_attn_cam(cam1)
    # 通过注意力下降层进一步传播 cam1 的相关性
    cam1 = self.attn_drop.relprop(cam1, **kwargs)
    # 通过 softmax 层传播相关性
    cam1 = self.softmax.relprop(cam1, **kwargs)
    # A = Q*K^T，计算查询和键的相关性
    (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
    cam_q /= 2  # 将 cam_q 的相关性除以 2
    cam_k /= 2  # 将 cam_k 的相关性除以 2
    # 将查询、键和值的相关性图重新排列为原始的 QKV 格式
    cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
    # 返回通过 QKV 层传播的相关性
    return self.qkv.relprop(cam_qkv, **kwargs)
```