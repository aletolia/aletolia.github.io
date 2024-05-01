# Visualization Work Flow

- 该工作流以 CLAM 的 heat map 生成为例

首先，CLAM 的输入是 WSI ，通常像素在千万级别，因此无论是训练，微调以及最后的可视化，都无法直接在原本图像上进行，因此总体流程大致如下：

1. 先将 WSI 分割为适合预训练模型输入尺寸的大小（例如 ResNet 是 256 × 256，ViT 一类的则是 224 × 224）
2. 接下根据任务目标进行微调，在 CLAM 的例子中是多实例学习（MIL Multiple Instance Learning），CLAM 发布之初，使用的是 ResNet50 作为迁移学习的预训练权重，这是一个 CNN ，那么如何做到 attention heat map visualization？
3. 实际上只是使用 ResNet50 作为 backbone，MIL 则是使用了 CLAM_sb 作为 neck 进行分类，因此其中的 attention 实际上是 neck 中的
4. 因此问题可以转换为：
   - 在模型的 fine-tune 和 eval 阶段过程中，如何将输入的 224 × 224 image 的 attention score 随训练一起导出
   - 如何将这个 attention score 叠加到对应的 image 上，生成一张类似于以往 attention visualization 的 heat map
   - 由于每次输入的 image 实际上是整张 WSI 很小的一部分，如何将这些 heat map 拼接起来重建成一张完整的 WSI heat map？
5. 另外改进方式也可以思考：相比于直接导出 attention score ，是否可以改用其他的解释手段，是否会比直接叠加更具有可解释性？

- [ ] TODO：尝试加入 attention_branch_network 的可视化手段

下面是对 `create_heatmaps.py` 的一些笔记

```python
def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    # 将输入特征转移到设定的计算设备上（如GPU）
    features = features.to(device)
    
    # 开启PyTorch的推理模式，这样可以提高性能并减少内存使用
    with torch.inference_mode():
        # 检查模型类型是否为CLAM_SB或CLAM_MB
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            # 执行模型推理
            model_results_dict = model(features)
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()  # 获取预测的类别ID

            # 如果模型是CLAM_MB类型，特别处理A（注意力权重）
            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]  # 选择对应预测类别的注意力权重

            # 将注意力权重数组转换为一维并移至CPU，再转换为NumPy数组
            A = A.view(-1, 1).cpu().numpy()

        else:
            # 如果模型不是CLAM_SB或CLAM_MB类型，抛出未实现错误
            raise NotImplementedError

        # 打印预测的类别、真实标签和每个类别的概率
        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(
            reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))
        
        # 获取概率最高的k个预测结果
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()  # 转换为NumPy数组
        ids = ids[-1].cpu().numpy()  # 转换为NumPy数组
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])  # 根据ID获取预测的类别名称

    # 返回类别ID、预测类别名称、概率和注意力权重
    return ids, preds_str, probs, A
```

定义了一个推断单个 WSI 的函数，输入 WSI 经特征提取后的特征文件，使用 CLAM_SB/MB 进行分类，可以看到 CLAM_SB 模型在定义时本身就可以返回 attention score：

```python
 logits, Y_prob, Y_hat, A, _ = model(features)
```

并且从张量转换成了 Numpy 数组：

```python
 A = A.view(-1, 1).cpu().numpy()
```

后面的内容按下不表，主要是输出模型的分类结果，并且返回了一个 Numpy 形式的 attention weights：

```python
return ids, preds_str, probs, A
```

接下来看执行程序：

```python
    # 计算patch大小和步长
    patch_size = tuple([patch_args.patch_size for i in range(2)])  # 创建patch尺寸的元组，通常是正方形
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))  # 根据重叠率计算步长
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
        patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))  # 打印相关信息
```

patch_size 在配置文件中大小为 256 × 256 ，是因为 CLAM 此前使用 ResNet50 来提取特征，这里创建了一个 256 × 256 维度的元组

```python
>>> (256, 256)
```

```python
    # 根据是否有特定的处理列表来加载和初始化幻灯片列表
    if data_args.process_list is None:
        # 检查data_dir是否为列表，即是否有多个数据目录
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                # 从每个目录中收集幻灯片文件
                slides.extend(os.listdir(data_dir))
        else:
            # 如果只有一个目录，则从该目录获取幻灯片列表，并排序
            slides = sorted(os.listdir(data_args.data_dir))
        # 过滤出指定扩展名的幻灯片
        slides = [slide for slide in slides if data_args.slide_ext in slide]
```

根据参数(`.svs`)，从目录中找出一次工作需要进行可视化的 WSI，这里出现了另一个函数 `initialize_df`

```python
def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
                  use_heatmap_args=False, save_patches=False):
    """
    初始化一个用于处理幻灯片图像的DataFrame，并设置默认的处理参数。
    
    参数:
        slides (DataFrame 或 类数组):
            如果是DataFrame，假设 'slide_id' 列包含幻灯片ID。
            如果是类数组，直接包含幻灯片ID列表。
        seg_params (字典): 分割参数，包括阈值和标志。
        filter_params (字典): 过滤参数，包括属性和阈值。
        vis_params (字典): 可视化参数，如显示级别和线条厚度。
        patch_params (字典): 打补丁参数，包括填充和轮廓函数。
        use_heatmap_args (布尔值): 是否包括与热图相关的参数（例如ROI坐标）。
        save_patches (布尔值): 是否包括保存补丁的参数。
    
    返回:
        DataFrame: 初始化后的DataFrame，列设置用于处理参数。
    """

    # 确定幻灯片的总数
    total = len(slides)
    
    # 从DataFrame中提取幻灯片ID，或使用提供的类数组ID
    if isinstance(slides, pd.DataFrame):
        slide_ids = slides['slide_id'].values
    else:
        slide_ids = slides

    # 使用numpy数组设置DataFrame的默认数据
    default_df_dict = {
        'slide_id': slide_ids,
        'process': np.full(total, 1, dtype=np.uint8)  # 默认处理标志设置为1
    }

    # 如果使用热图参数，初始化标签列的默认值
    if use_heatmap_args:
        default_df_dict.update({
            'label': np.full(total, -1)  # 默认标签设置为-1
        })
    
    # 更新DataFrame字典，包括分割、过滤和可视化参数
    default_df_dict.update({
        'status': np.full(total, 'tbp'),  # 状态设置为'待处理'
        'seg_level': np.full(total, int(seg_params['seg_level']), dtype=np.int8),
        'sthresh': np.full(total, int(seg_params['sthresh']), dtype=np.uint8),
        'mthresh': np.full(total, int(seg_params['mthresh']), dtype=np.uint8),
        'close': np.full(total, int(seg_params['close']), dtype=np.uint32),
        'use_otsu': np.full(total, bool(seg_params['use_otsu']), dtype=bool),
        'keep_ids': np.full(total, seg_params['keep_ids']),
        'exclude_ids': np.full(total, seg_params['exclude_ids']),
        'a_t': np.full(total, int(filter_params['a_t']), dtype=np.float32),
        'a_h': np.full(total, int(filter_params['a_h']), dtype=np.float32),
        'max_n_holes': np.full(total, int(filter_params['max_n_holes']), dtype=np.uint32),
        'vis_level': np.full(total, int(vis_params['vis_level']), dtype=np.int8),
        'line_thickness': np.full(total, int(vis_params['line_thickness']), dtype=np.uint32),
        'use_padding': np.full(total, bool(patch_params['use_padding']), dtype=bool),
        'contour_fn': np.full(total, patch_params['contour_fn'])
    })

    # 如果保存补丁，添加与补丁阈值相关的参数
    if save_patches:
        default_df_dict.update({
            'white_thresh': np.full(total, int(patch_params['white_thresh']), dtype=np.uint8),
            'black_thresh': np.full(total, int(patch_params['black_thresh']), dtype=np.uint8)
        })

    # 对于热图参数，初始化坐标列为NaN
    if use_heatmap_args:
        default_df_dict.update({
            'x1': np.empty(total).fill(np.NaN), 
            'x2': np.empty(total).fill(np.NaN), 
            'y1': np.empty(total).fill(np.NaN), 
            'y2': np.empty(total).fill(np.NaN)
        })

    # 如果幻灯片是一个DataFrame，更新它的默认值或插入新列
    if isinstance(slides, pd.DataFrame):
        temp_copy = pd.DataFrame(default_df_dict)  # 临时DataFrame，包含默认参数
        for key in default_df_dict.keys():
            if key in slides.columns:
                mask = slides[key].isna()  # 在现有列中找到缺失值的掩码
                slides.loc[mask, key] = temp_copy.loc[mask, key]  # 填充缺失值
            else:
                slides.insert(len(slides.columns), key, default_df_dict[key])  # 插入新列
    else:
        # 将类数组的幻灯片ID转换为DataFrame
        slides = pd.DataFrame(default_df_dict)
    
    return slides
```

首先 `slides` 变量是一个列表，具体来说，根据配置文件中的 `data_dir` 地址，读取其中后缀名为 `.svs` 的文件，然后放入 `slides` 中

```python
    # 根据是否有特定的处理列表来加载和初始化幻灯片列表
    if data_args.process_list is None:
        # 检查data_dir是否为列表，即是否有多个数据目录
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                # 从每个目录中收集幻灯片文件
                slides.extend(os.listdir(data_dir))
        else:
            # 如果只有一个目录，则从该目录获取幻灯片列表，并排序
            slides = sorted(os.listdir(data_args.data_dir))
        # 过滤出指定扩展名的幻灯片
        slides = [slide for slide in slides if data_args.slide_ext in slide]
```

计算 `slides` 中 WSI 的总数，并提取文件名

```python
    # 确定幻灯片的总数
    total = len(slides)
    
    # 从DataFrame中提取幻灯片ID，或使用提供的类数组ID
    if isinstance(slides, pd.DataFrame):
        slide_ids = slides['slide_id'].values
    else:
        slide_ids = slides
```

接着将 `slides` 转换为字典 `df`

```python
df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
```

加载权重和模型

```python
ckpt_path = model_args.ckpt_path
if model_args.initiate_fn == 'initiate_model':
    model = initiate_model(model_args, ckpt_path)
```

其中

```python
def initiate_model(args, ckpt_path, device='cuda'):
  """
  初始化模型函数

  参数:
      args: 命令行参数 (argparse.Namespace 对象)
          - args.drop_out: dropout 衰减率 (来自命令行参数)
          - args.n_classes: 分类类别数量 (来自命令行参数)
          - args.embed_dim: 词嵌入维度 (来自命令行参数)
          - args.model_size: 模型尺寸 (来自命令行参数，可能为 None)
          - args.model_type: 模型类型 (来自命令行参数，例如 'clam_sb', 'clam_mb', 'mil')
      ckpt_path: 预训练权重文件路径 (字符串)
      device: 训练设备 ('cuda' 或 'cpu')，默认为 'cuda'

  返回:
      model: 加载预训练权重的模型对象
  """

  print('初始化模型')  # 打印信息，表示开始初始化模型

  # 创建模型参数字典
  model_dict = {
      "dropout": args.drop_out,  # 加入 dropout 衰减率
      'n_classes': args.n_classes,  # 指定分类类别数量
      "embed_dim": args.embed_dim   # 设置词嵌入维度
  }

  # 根据模型尺寸和类型更新模型参数字典 (仅适用于 clam_sb 和 clam_mb 模型)
  if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
    model_dict.update({"size_arg": args.model_size})  # 更新字典，加入 'size_arg' 参数

  # 根据模型类型创建模型对象
  if args.model_type =='clam_sb':
    model = CLAM_SB(**model_dict)  # 使用 **model_dict 参数字典初始化 CLAM_SB 模型
  elif args.model_type =='clam_mb':
    model = CLAM_MB(**model_dict)  # 使用 **model_dict 参数字典初始化 CLAM_MB 模型
  else:  # 对于 MIL 模型
    if args.n_classes > 2:
      model = MIL_fc_mc(**model_dict)  # 多分类 (>2) 的 MIL 模型，使用 MIL_fc_mc 类
    else:
      model = MIL_fc(**model_dict)   # 二分类 (<=2) 的 MIL 模型，使用 MIL_fc 类

  # 打印模型结构 (可能需要额外的 print_network 函数实现)
  print_network(model)

  # 加载预训练权重
  ckpt = torch.load(ckpt_path)

  # 清理检查点字典，去除不需要的键 (例如 'instance_loss_fn')
  ckpt_clean = {}
  for key in ckpt.keys():
    if 'instance_loss_fn' in key:
      continue
    ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
  model.load_state_dict(ckpt_clean, strict=True)

  # 将模型转移到指定设备 (cuda 或 cpu)
  model = model.to(device)

  # 设置模型为评估模式
  model.eval()

  return model
```

这里加载的是分类模型，即经过微调后的分类模型，接下来要加载特征提取框架

```python
feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
_ = feature_extractor.eval()  # 将特征提取器设置为评估模式
feature_extractor = feature_extractor.to(device)  # 将特征提取器移至设备（GPU或CPU）
print('Done!')
```

设置图像处理参数，注意左上角和右下角坐标的初始化

```python
blocky_wsi_kwargs = {
        'top_left': None,  # 左上角坐标，初始化为None
        'bot_right': None,  # 右下角坐标，初始化为None
        'patch_size': patch_size,  # 设置处理块的尺寸
        'step_size': patch_size,  # 设置步长为块的大小
        'custom_downsample': patch_args.custom_downsample,  # 设置自定义降采样率
        'level': patch_args.patch_level,  # 设置处理的图像层级
        'use_center_shift': heatmap_args.use_center_shift  # 是否使用中心偏移
    }
```

初始化 WSI

```python
wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
```

主要是将传入的 WSI 转化为 `WholeSlideImage` 对象，并且直接进行分割

```python
    # 使用给定的路径创建WholeSlideImage对象
    wsi_object = WholeSlideImage(wsi_path)

    # 如果分割参数中的分割层级为自动选择(-1)，则获取最佳下采样层级来优化分割效果
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    # 调用WSI对象的segmentTissue方法进行组织区域的分割，并传入分割和过滤器参数
    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
```

先进行下采样，接着进行分割

```python
class WholeSlideImage(object):  # 定义一个名为WholeSlideImage的类。
    def __init__(self, path):
        """
        初始化WholeSlideImage对象。
        
        Args:
            path (str): WSI文件的完整路径。
        """
        
        # 提取不含扩展名的文件名作为WSI的名称。
        self.name = os.path.splitext(os.path.basename(path))[0]
        # 使用OpenSlide打开指定的全扫描图像。
        self.wsi = openslide.open_slide(path)
        # 验证不同级别的下采样因子是否一致。
        self.level_downsamples = self._assertLevelDownsamples()
        # 存储不同放大级别的图像维度。
        self.level_dim = self.wsi.level_dimensions
        
        # 初始化用于保存组织和肿瘤轮廓的占位属性。
        self.contours_tissue = None
        self.contours_tumor = None
        # 初始化用于后续可能需要的HDF5文件的占位属性。
        self.hdf5_file = None
```

接着来看定义，首先读取一个 WSI 文件，然后初始化组织和肿瘤的轮廓（事先声明变量）

```python
def initSegmentation(self, mask_file):
        # 从pickle文件中加载分割结果
        import pickle
        asset_dict = load_pkl(mask_file)
        # 从加载的字典中提取组织孔洞和组织轮廓信息
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']
```

```python
def saveSegmentation(self, mask_file):
        # 将分割结果封装到字典中
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        # 使用pickle保存分割结果到文件
        save_pkl(mask_file, asset_dict)
```

分割的 I/O

接下来是分割函数：

```python
def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False, 
                            filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """
            通过HSV转换 -> 中值滤波 -> 二值化阈值来分割组织。
        """
```

输入的参数：

1. 分割的级别（下采样水平）默认为 0

2. 放大阈值（默认为 20 ×）

3. 分割大小：255+1

   下面是图像的预处理

```python
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # 将RGB图像转换为HSV色彩空间
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # 对HSV中的S通道应用中值滤波
```

首先将在特定的分割层级读取 WSI ，然后将其转换成 Numpy 数组，其中左上角是 （0，0），并转换为 HSV 颜色通道，然后进行中值滤波

接着使用 Otsu 法进行分割

```python
if use_otsu:
            # 如果使用Otsu算法自动找到最优阈值进行二值化
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
else:
            # 使用固定阈值进行二值化
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

if close > 0:
            kernel = np.ones((close, close), np.uint8)  # 创建一个正方形的结构元素
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)  # 对二值图像应用形态学闭运算
```

然后计算每个像素大小在 WSI 上的实际面积大小

```python
scale = self.level_downsamples[seg_level]  # 获取当前分割级别的下采样因子
scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))  # 根据下采样因子调整参考补丁大小
filter_params['a_t'] *= scaled_ref_patch_area  # 调整过滤参数中的面积阈值
filter_params['a_h'] *= scaled_ref_patch_area  # 调整过滤参数中的洞面积阈值
```

这个函数返回一个数组（包含孔洞坐标的 Numpy 数组）以及一个包含轮廓层次信息（描述了轮廓之间的关系，例如父子关系和嵌套级别）的可选数组

```python
contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
```

其中：

mode: 轮廓检索模式。它指定如何检索图像中的轮廓。常用模式包括：

- cv2.RETR_EXTERNAL: 只检索外部轮廓，即图像边界。
- cv2.RETR_LIST: 检索所有轮廓，但不建立层次关系。
- cv2.RETR_CCOMP: 检索所有轮廓，并建立两个层次：外部轮廓和内孔轮廓。

method: 轮廓近似方法。它指定如何简化轮廓。常用方法包括：

- cv2.CHAIN_APPROX_NONE: 保存所有轮廓点。
- cv2.CHAIN_APPROX_SIMPLE: 使用折线近似轮廓。
- cv2.CHAIN_APPROX_TC89_KCOS: 使用 Douglas-Peucker 算法简化轮廓。

在此模式下，`hierarchy` 是一个 2D 数组，其中第一行代表外部轮廓，后续行代表嵌套在前一行轮廓中的轮廓，因此，如果发现其中存在嵌套，则需要保留内部的孔洞（即 `hierarchy[0]`），如果 `len(hierarchy) != 2`，则表明 `hierarchy` 数组是 1D 的。在这种情况下，`squeeze()` 函数用于删除任何不必要的维度，有效地将 1D 数组转换为单行数组

```python
hierarchy = hierarchy[0] if len(hierarchy) == 2 else hierarchy.squeeze()
```

接着根据面积过滤轮廓，并返回过滤后的前景轮廓和洞轮廓

```python
 def _filter_contours(contours, hierarchy, filter_params):
            """
                根据面积过滤轮廓。
            """
            filtered = []  # 初始化一个空列表，用来存储过滤后的轮廓索引
            # 找到所有前景轮廓的索引（父级为-1）
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []  # 初始化一个空列表，用来存储所有洞的索引
            
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]  # 获取当前前景轮廓
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)  # 获取属于当前轮廓的洞的索引
                a = cv2.contourArea(cont)  # 计算当前轮廓的面积
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]  # 计算每个洞的面积
                a -= np.sum(hole_areas)  # 从前景轮廓面积中减去所有洞的面积
                if a == 0: continue  # 如果轮廓面积减去洞面积后为0，则跳过
                # 如果轮廓面积大于参数中设置的面积阈值，添加到filtered列表
                if a > filter_params['a_t']: 
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]  # 获取所有过滤后的前景轮廓
            
            hole_contours = []  # 初始化一个空列表，用来存储过滤后的洞的轮廓
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]  # 获取未过滤的洞的轮廓
                unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)  # 按面积降序排序
                # 根据面积过滤，并只保留最大的n个洞的轮廓，n由max_n_holes参数决定
                filtered_holes = [hole for hole in unfiltered_holes if cv2.contourArea(hole) > filter_params['a_h']]
                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours
```

首先初始化一个空列表，用来存储过滤后的轮廓索引，在 `hierarchy` 中，父轮廓的值为 -1，因此取出这个数组的第一列和 -1 比较，仅保留（返回） `hierarchy` 数组第二列中所有等于 -1 的元素的索引

在循环中，首先依次读取当前前景轮廓及其中洞的索引

```python
cont = contours[cont_idx]  # 获取当前前景轮廓
holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)  # 获取属于当前轮廓的洞的索引
```

计算整个前景的面积后减去其中洞的面积

```python
# 如果轮廓面积大于参数中设置的面积阈值，添加到filtered列表
if a > filter_params['a_t']: 
filtered.append(cont_idx)
all_holes.append(holes)
```

汇总除去孔洞之后的前景

```python
foreground_contours = [contours[cont_idx] for cont_idx in filtered]  # 获取所有过滤后的前景轮廓
```

现在我们有了三个变量

- `filtered`：一个列表，分别包含所有过滤后的前景
- `holes`：一个列表，包含所有孔洞
- `foreground_contour`：汇总后的前景

接下来决定保留多少大的孔

```python
for cont_idx in hierarchy_1:
                cont = contours[cont_idx]  # 获取当前前景轮廓
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)  # 获取属于当前轮廓的洞的索引
                a = cv2.contourArea(cont)  # 计算当前轮廓的面积
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]  # 计算每个洞的面积
                a -= np.sum(hole_areas)  # 从前景轮廓面积中减去所有洞的面积
                if a == 0: continue  # 如果轮廓面积减去洞面积后为0，则跳过
                # 如果轮廓面积大于参数中设置的面积阈值，添加到filtered列表
                if a > filter_params['a_t']: 
                    filtered.append(cont_idx)
                    all_holes.append(holes)
```

最后返回经过汇总的前景以及相应的孔洞

接着调整大小，以符合比例尺

```python
# 调整轮廓到全图尺寸
self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
self.holes_tissue = self.scaleHolesDim(hole_contours, scale)
```

相当于完成了组织分割

```python
wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
```

接下来计算实际可视化时的图块大小，根据 WSI 的下采样水平，计算实际的尺寸，以及一些前期准备（如果还没有提取过特征则进行特征提取）

```python
        # 计算实际用于可视化的图块大小
        # 获取WSI对象中指定层级的下采样率
        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        # 计算可视化时使用的实际图块大小，考虑了下采样率和自定义的降采样因子
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        # 设置保存块映射和掩码的文件路径
        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        # 如果可视化参数中指定的层级小于0，则自动选择最佳的层级用于下采样
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        # 使用WSI对象的visWSI方法进行可视化，生成掩码，并保存掩码图片
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)

        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')

       # 检查HDF5格式特征文件是否已存在，若不存在，则计算特征并保存
        if not os.path.isfile(h5_path):
            # 计算WSI图像的特征并将结果保存到指定路径
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                                    model=model, 
                                                    feature_extractor=feature_extractor, 
                                                    img_transforms=img_transforms,
                                                    batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
                                                    attn_save_path=None, feat_save_path=h5_path, 
                                                    ref_scores=None)

        # 检查PyTorch格式特征文件是否已存在，若不存在，则从HDF5文件加载特征并保存为PyTorch格式
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")  # 打开HDF5文件
            features = torch.tensor(file['features'][:])  # 读取特征数据并转换为PyTorch张量
            torch.save(features, features_path)  # 将特征保存为PyTorch文件
            file.close()  # 关闭HDF5文件
```

推断

```python
Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
```

读取坐标：

```python
if not os.path.isfile(block_map_save_path):
            file = h5py.File(h5_path, "r")  # 打开HDF5文件读取坐标
            coords = file['coords'][:]  # 读取坐标
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}  # 创建包含注意力分数和坐标的字典
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')  # 保存数据到HDF5文件
```

注意这里创建的字典

```python
asset_dict = {'attention_scores': A, 'coords': coords}  # 创建包含注意力分数和坐标的字典
```

读取时一一对应

```python
coord_dset = file['coords']  # 访问文件中的坐标数据集
scores = dset[:]  # 将注意力分数数据读取到内存中
coords = coord_dset[:]  # 将坐标数据读取到内存中
```

接下来设置如何取样，配置文件中，sample 是一个字典

```python
  samples:
    - name: "topk_high_attention"
      sample: true
      seed: 1
      k: 15 # save top-k patches
      mode: topk
```

创建一个标签 `tag`，用于标识该样本的取样结果。标签格式为 `"label_{}_pred_{}"`，其中 `label` 是样本的实际标签，`Y_hats[0]` 是样本的预测标签

然后创建一个目录 `sample_save_dir`，用于保存该样本的取样图像块。目录结构为 `exp_args.production_save_dir/exp_args.save_exp_code/sampled_patches`

```python
        for sample in samples:
            if sample['sample']:  # 检查是否需要进行取样
                # 为每个取样创建一个标签，包括实际标签和预测标签
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                # 创建保存取样图像块的目录
                sample_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)  # 确保目录存在
                print('sampling {}'.format(sample['name']))  # 打印正在取样的信息
```

根据坐标的上下限，首先将注意力分数转换为百分位数，再根据给定的坐标范围筛选分数和坐标

调用示意：

```python
sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                                             score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
```

```python
def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):
    """
    根据指定的模式从分数列表中抽样区域兴趣（ROI）。

    参数:
    scores (array-like): 与坐标关联的分数数组。
    coords (array-like): 对应的坐标数组。
    k (int): 抽样数量，默认为 5。
    mode (str): 抽样模式，可以是 'range_sample'（区间抽样），'topk'（最高分抽样），或 'reverse_topk'（最低分抽样）。
    seed (int): 随机种子，默认为 1。
    score_start (float): 分数区间的开始值，默认为 0.45。
    score_end (float): 分数区间的结束值，默认为 0.55。
    top_left (tuple): 抽样区域的左上角坐标，如果为 None，则不限制区域。
    bot_right (tuple): 抽样区域的右下角坐标，如果为 None，则不限制区域。
    """
    # 检查分数数组的维度，如果是二维的，则将其平展为一维数组
    if len(scores.shape) == 2:
        scores = scores.flatten()  # 将分数数组平展为一维
    
    # 将分数转换为百分位数
    scores = to_percentiles(scores)
    
    # 如果提供了坐标范围的上限和下限，则根据这些范围过滤分数和坐标
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
    
    # 根据选择的模式来抽样索引
    if mode == 'range_sample':
        # 如果模式是区间抽样，则调用 sample_indices 函数
        # 该函数根据分数的百分位数区间来抽样
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        # 如果模式是选择最高的 k 个分数，则调用 top_k 函数
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        # 如果模式是选择最低的 k 个分数，则调用 top_k 函数，并设置 invert 参数为 True
        sampled_ids = top_k(scores, k, invert=True)
    else:
        # 如果提供了未定义的模式，则抛出异常
        raise NotImplementedError
    
    coords = coords[sampled_ids]  # 获取抽样的坐标
    scores = scores[sampled_ids]  # 获取抽样的分数

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset
```

```python
def screen_coords(scores, coords, top_left, bot_right):
    """
    根据给定的坐标范围筛选分数和坐标。

    参数:
    scores (array-like): 与 coords 关联的分数数组。
    coords (array-like): 坐标数组。
    top_left (tuple): 筛选范围的左上角坐标。
    bot_right (tuple): 筛选范围的右下角坐标。
    """
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    # 创建一个布尔数组，标记在指定坐标范围内的坐标
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]  # 筛选出范围内的分数
    coords = coords[mask]  # 筛选出范围内的坐标
    return scores, coords
```

返回了一个字典，即对应的坐标以及对应的注意力分数

```python
asset = {'sampled_coords': coords, 'sampled_scores': scores}
```

依次读取，并且保存

```python
# 遍历取样结果并保存
for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'],sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))  # 打印每个取样的坐标和分数
                    # 从WSI中读取相应的图像区域
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    # 保存图像块
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))
```

这里首先使用循环依次调用坐标及其对应的 attention score，然后读取相应区域

```python
patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
```

Openslide 的 `read_region` 函数接受坐标（元组），采样等级和大小三个参数，然后返回一个 PIL.Image ，这里还进行了向 RGB 的转换

在读取图像，并且注意力分数已经一一对应的准备下，接下来进行可视化工作

```python
heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, 
                                  alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, 
                                  blank_canvas=False, thresh=-1, patch_size=vis_patch_size, 
                                  convert_to_percentiles=True)
```

`drawHeatmap` 接受一个 attention score 列表以及对应的坐标列表，WSI 图片的路径或者 `WholeSlideImage` 对象，返回一个 PIL.Image，具体来说，是调用了 `WholeSlideImage` 的 `visHeatmap` 方法

```python
heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
```

首先确认下采样级别，然后获取对应级别的下采样比例

```python
# 确定可视化的级别
if vis_level < 0:
   vis_level = self.wsi.get_best_level_for_downsample(32)

# 获取对应级别的下采样比例
downsample = self.level_downsamples[vis_level]
scale = [1/downsample[0], 1/downsample[1]]  # 从0级到目标级别的缩放比例
```

将 attention score 转换为一维数组

```python
if len(scores.shape) == 2:
           scores = scores.flatten()
```

筛选坐标和分数，只包含指定区域内的

```python
if top_left is not None and bot_right is not None:
           # 如果指定了区域，首先筛选出该区域内的分数和坐标
           scores, coords = screen_coords(scores, coords, top_left, bot_right)
           # 调整坐标系统，使得左上角的坐标成为新的原点
           coords = coords - np.array(top_left)  
           # 计算指定区域的宽度和高度，考虑到了缩放比例
           w, h = np.array(bot_right) * np.array(scale) - np.array(top_left) * np.array(scale)
           # 设置热图的区域大小
           region_size = (w, h)
else:
           # 如果没有指定绘制区域，则默认使用全图的尺寸
           region_size = self.level_dim[vis_level]
           top_left = (0,0)  # 默认左上角坐标为(0,0)
           bot_right = self.level_dim[0]  # 右下角坐标为当前视觉层级的维度
           w, h = region_size  # 宽度和高度设为视觉层级的尺寸
```

其中，首先根据给定的坐标范围筛选分数和坐标

```python
def screen_coords(scores, coords, top_left, bot_right):
    """
    根据给定的坐标范围筛选分数和坐标。

    参数:
    scores (array-like): 与 coords 关联的分数数组。
    coords (array-like): 坐标数组。
    top_left (tuple): 筛选范围的左上角坐标。
    bot_right (tuple): 筛选范围的右下角坐标。
    """
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    # 创建一个布尔数组，标记在指定坐标范围内的坐标
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]  # 筛选出范围内的分数
    coords = coords[mask]  # 筛选出范围内的坐标
    return scores, coords
```

仅在这个函数中对画布进行初始化

```python
 coords = coords - np.array(top_left)
```

转换分数位数

```python
 # 将分数转换为百分位数
       if convert_to_percentiles:
           scores = to_percentiles(scores) 
       # 如果启用，将注意力分数转换为百分位数，有助于处理极端值和提高可视化的一致性

       scores /= 100  # 将分数标准化到[0, 1]区间
       # 这一步是为了简化后续处理，确保处理的分数都在一个标准的范围内
```

创建画布，其中 `np.full` 创建了一个具有指定形状和值的数组

```python
# 创建覆盖图和计数器，用于累积每个像素的注意力分数
       overlay = np.full(np.flip(region_size), 0).astype(float)
       # 初始化一个与热图区域大小相匹配的覆盖图，用于存储每个像素位置的累积分数
       counter = np.full(np.flip(region_size), 0).astype(np.uint16)
       # 初始化一个计数器矩阵，用于记录每个像素位置累积的次数，以便进行平均处理
```

接着检查每个分数是否大于或等于预设的阈值。对于大于等于阈值的分数，它在相应的补丁区域内将分数值累加到`overlay`数组中，并在`counter`数组中将相应的计数值加一

```python
for idx, (score, coord) in enumerate(zip(scores, coords)):
           if score >= threshold:
               # 只处理分数高于设定阈值的坐标
               overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
               # 在对应补丁的区域内累加分数
               counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1
               # 在对应补丁的区域内增加计数
```

将 `overlay` 中的分数除以 `counter` 数组中相应的计数值，从而计算出平均分数

```python
overlay[~(counter == 0)] /= counter[~(counter == 0)]
```

```python
 # 创建图像画布
       if not blank_canvas:
           # 如果不使用空白画布，则从WSI中读取相应区域的图像作为画布
           img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
       else:
           # 如果使用空白画布，则创建一个纯色的画布
           img = np.full(region_size[::-1], canvas_color, dtype=np.uint8)
```

生成热图

```python
heatmap = (cmap(overlay)[:, :, :3] * 255).astype(np.uint8)
```

1. **使用颜色映射（Color Map）**：
   - `cmap(overlay)`：这里的 `cmap` 代表一个颜色映射函数，这个函数接受 `overlay` 数组（其中包含每个像素的平均分数）并将这些数值映射到一个颜色空间，从而为每个数值分配一个具体的颜色。
   - 结果是一个三维数组，其中的每个元素都是表示RGB颜色的三个分量（红色、绿色、蓝色）加上一个可选的透明度分量（Alpha）。

2. **选择颜色通道并转换**：
   - `[:, :, :3]`：这个切片操作选择了每个像素的RGB通道，忽略了Alpha通道
   - `* 255`：由于颜色映射函数默认输出的颜色分量范围通常在0到1之间，这一步将这些值缩放到0到255的范围，这是标准的8位颜色表示方法。

3. **转换为无符号8位整型**：
   - `.astype(np.uint8)`：最后，这个操作将数组中的元素类型转换为无符号8位整数（uint8）

叠加在原有的画布上

```python
img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
```

1. **函数 cv2.addWeighted 的参数**：
   - `img`：原始图像。
   - `1 - alpha`：原始图像的权重。这里 `alpha` 是一个介于 0 和 1 之间的值，表示热图的透明度。`1 - alpha` 表示原始图像的不透明度，即原始图像在合成图像中的可见度。
2. **图像叠加的效果**：
   - `cv2.addWeighted` 函数计算公式为：`dst = src1 * alpha + src2 * beta + gamma`。在这种情况下，`src1` 是 `img`，`src2` 是 `heatmap`，`gamma` 是 0。
   - 这个操作的结果是一个新图像 `img`，其中原始图像和热图按照指定的权重叠加在一起。通过调整 `alpha` 的值，可以控制热图覆盖的透明度，从而在保持原始图像可见的同时，叠加上热图的信息。

# Segmentation Work Flow

这里简化了描述，仅仅大致讲一下流程

首先这个任务可以被拆分为几个任务

1. 读取文件，从 csv 文件的特定列读取处理列表，为了生成进度条，也需要初始化一下处理总数
2. 能够迭代的进行分割
3. 并保存相应的坐标

根据上面的经验，在初始化 `WholeSlideImage` 对象后，这个对象本身就有一个 `segmentTissue` 方法，因此我们要做的就是确定这个方法的参数，也就是采样层级

和之前一样，只需要调用 Openslide 本身的方法即可

```python
best_level = wsi.get_best_level_for_downsample(64)
```

以及一些鲁棒性检查（此处略）

但由于这里是分割流程，我们还需要调用 `patching` 方法，这里详细解释一下其实现（在调用 `patching` 以前已经调用过了 `segmentTissue` 方法，因此此时的 `WholeSlideImage` 对象已经包含了完整的 `contours_tissue` 和 `holes_tissue` 属性，因此直接开始迭代去除了孔洞的组织轮廓：

```python
for idx, cont in enumerate(self.contours_tissue):
```

这里继续调用了另一个方法 `process_contour`

```python
asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs)
```

首先初始化边界：

```python
start_x, start_y, w, h = cv2.boundingRect(cont)
```

接着计算 patch 的实际尺寸，这和 WSI 的下采样系数有关，在不同的下采样系数下，相同像素尺寸的图像在实际图像中的面积大小不同，这主要是由于以 svs 格式为代表的 WSI 是一种多层次图像，可以在不同的放大倍数下查看

```
patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])
```

乘以下采样率后，`ref_patch_size` 代表在当前层级上调整后的补丁大小。这意味着，如果原始补丁大小为 256x256 像素，而下采样率为 4x4，那么参考补丁大小将是 1024x1024 像素

在指定轮廓检测方法以后，生成 X 和 Y 方向的坐标范围

```
x_range = np.arange(start_x, stop_x, step=step_size_x)  # 从起始X坐标到终止X坐标，按步进尺寸生成坐标点
y_range = np.arange(start_y, stop_y, step=step_size_y)  # 从起始Y坐标到终止Y坐标，同样按步进尺寸生成坐标点
x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')  # 使用网格生成方法，创建一个坐标网格
coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()  # 将坐标网格扁平化并转置，生成坐标点对的列表
```

- `x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')`：使用 `numpy` 的 `meshgrid` 函数生成一个二维的坐标网格。这个函数默认是`xy`（笛卡尔）索引，但在这里使用了 `ij`（矩阵）索引，意味着第一个维度是x坐标，第二个维度是y坐标。结果是两个数组 `x_coords` 和 `y_coords`，它们的每个元素对应网格中的一个坐标点的 x 和 y 值

- `ccoord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()`：这一步首先将 `x_coords` 和 `y_coords` 数组扁平化，将二维网格转换为一维数组。然后，将这两个数组合并并转置，生成一个包含所有坐标对的二维数组。每行代表一个坐标点，第一个元素是 x 坐标，第二个元素是 y 坐标

接下来根据 CPU 的核心数开始并行处理，调用了 `WholeSlideImage` 对象中的 `process_coord_candidate` 方法，这是一个静态方法，可以直接通过类调用，而无需创建一个 `WholeSlideImage` 的实例

```python
    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None
```

调用了另一个静态方法 `isInContours`

```python
    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        """
        检查一个点是否位于轮廓内，并且不位于任何关联的“洞”内。

        参数:
            cont_check_fn (function): 用于检查点是否在主轮廓内的函数。
            pt (tuple): 需要检查的点的坐标。
            holes (list of np.array, optional): 关联的“洞”的轮廓列表。
            patch_size (int, optional): 补丁的大小，用于确定“洞”检查中点的中心位置。

        返回:
            int: 如果点在轮廓内且不在“洞”内，返回1，否则返回0。
        """
        if cont_check_fn(pt):  # 检查点是否在主轮廓内
            if holes is not None:
                # 如果存在“洞”，确保点不在任何“洞”内
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1  # 没有“洞”，点在轮廓内
        return 0  # 点不在轮廓内
```

函数检查 `coord` 是否位于由 `contour_holes` 定义的轮廓内，同时考虑了 `ref_patch_size`。即以 `coord` 为中心，大小为 `ref_patch_size` 的矩形区域是否完全或部分位于轮廓内

这一段主要是从 `segmentTissue` 方法返回的坐标中进行二次过滤

```python
results = np.array([result for result in results if result is not None])
```

- 使用列表推导式过滤掉结果中的 `None` 值。`None` 表示坐标点不满足处理条件（不在特定轮廓内）
- 将过滤后的结果转换为 NumPy 数组，方便后续处理

然后组装成字典

```python
if len(results) > 0:
            asset_dict = {'coords' : results}
            attr = {'patch_size' : patch_size,
                    'patch_level' : patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim': self.level_dim[patch_level],
                    'name': self.name,
                    'save_path': save_path}
```

1. **组装坐标数据**：
   - `asset_dict = {'coords': results}`: 创建了一个字典 `asset_dict`，其中包含一个键 `coords`，值为过滤后的有效结果 `results`。
2. **组装属性数据**：
   - `patch_size`: 表示处理过程中使用的补丁大小
   - `patch_level`: 表示使用的图像层级
   - `downsample`: 从类属性 `self.level_downsamples` 中获取当前层级的下采样率
   - `downsampled_level_dim`: 将类属性 `self.level_dim` 中对应层级的尺寸转换为元组形式
   - `level_dim`: 直接从 `self.level_dim` 获取当前层级的维度数据
   - `save_path`: 表示数据保存路径

最后分装并返回

```python
attr_dict = {'coords' : attr}
return asset_dict, attr_dict
```

由于最后也需要返回图像，因此还需要把图像块拼起来

```python
def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    """
    使用来自 HDF5 文件的坐标在 WSI 对象的指定级别上拼接图像块。

    参数:
    hdf5_file_path (str): 包含坐标的 HDF5 文件路径。
    wsi_object (object): 一个包含整体扫描图像（WSI）数据和方法的对象。
    downscale (int): 输出图像尺寸的降低因子，以便于更容易地处理和可视化。
    draw_grid (bool): 是否在每个图像块周围绘制网格。
    bg_color (tuple): 背景颜色，默认为黑色。
    alpha (float): 背景的透明度，如果设置为 -1，则使用不透明的 RGB 模式；否则使用带有透明度的 RGBA 模式。
    """
```

仍然和之前一样，需要先获得下采样界别

```python
    # 获取 OpenSlide 对象并读取最顶层的维度
    wsi = wsi_object.getOpenSlide()  # 调用 WSI 对象的方法获取 OpenSlide 对象
    w, h = wsi.level_dimensions[0]  # 获取最顶层的图像维度，即是最高分辨率的图像层
    print('原始大小: {} x {}'.format(w, h))  # 打印原始图像的宽度和高度
    
    # 根据降采样倍数获取最佳的级别和对应的维度
    vis_level = wsi.get_best_level_for_downsample(downscale)  # 获取给定降采样倍数下的最佳图像级别
    w, h = wsi.level_dimensions[vis_level]  # 获取该级别的维度，这是根据降采样调整后的尺寸
    print('用于拼接的降采样大小: {} x {}'.format(w, h))  # 打印用于拼接的降采样后的图像大小
```

读取坐标，由于 HDF5 中储存的是字典，键值配对即可

```python
    # 从 HDF5 文件中读取坐标
    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['coords']  # 获取存储坐标的数据集
        coords = dset[:]  # 读取所有坐标
        print('开始拼接 {}'.format(dset.attrs['name']))  # 打印开始拼接的提示，包含数据集的名称
        patch_size = dset.attrs['patch_size']  # 读取图像块的尺寸
        patch_level = dset.attrs['patch_level']  # 读取图像块的级别
```

像之前一样调整 patch 的大小

```python
# 调整图像块大小以适应 WSI 的缩放级别
patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
```

创建画布，使用 Pillow 包

```python
heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
```

将画布转换为数组

```python
heatmap = np.array(heatmap)
```

之后调用了方法 `DrawMapFromCoords`

```python
def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=True):
    """
    基于WSI对象的坐标在画布上绘制图像块，这些坐标来自于特定的可视化级别。

    参数:
    canvas (array): 图像块将被绘制的画布数组。
    wsi_object (WSI对象): 包含WSI数据和方法的整体扫描图像对象。
    coords (列表，元素为元组): 应该从中提取图像块的坐标。
    patch_size (元组): 需要提取并绘制的图像块大小。
    vis_level (整数): 应该从中提取图像块的WSI金字塔级别。
    indices (列表，元素为整数，可选): 使用的坐标索引，默认使用所有坐标。
    draw_grid (布尔值，可选): 是否在图像块周围绘制网格线，默认为真。
    """
```

遍历坐标，在每个循环中，将每个坐标代表的位置和相应区域转换为 RGB ，并放到相应位置

```python
# 遍历所有索引以绘制每个图像块
for idx in tqdm(range(total)):        
    patch_id = indices[idx]
    coord = coords[patch_id]
    # 从WSI读取相应位置和级别的区域，转换为RGB
    patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
    # 根据降采样因子调整坐标
    coord = np.ceil(coord / downsamples).astype(np.int32)
    # 计算画布上对应位置的实际可用形状
    canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
    # 将调整尺寸后的图像块放置到画布的对应位置
    canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
    # 如果设置为绘制网格，则调用DrawGrid函数
    if draw_grid:
        DrawGrid(canvas, coord, patch_size)
```

其中需要注意的是

```
canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
```

- `coord[1]:coord[1]+patch_size[1]` 和 `coord[0]:coord[0]+patch_size[0]`：这两个切片操作确定了从 `canvas` 中抽取的矩形区域的纵向和横向范围。`coord` 代表起始点的y坐标和x坐标。`patch_size` 表示要抽取的矩形区域的高度和宽度

- `:3`：这个切片操作选择了前三个通道

以及粘贴操作

```python
canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
```

1. **选择 `canvas` 的目标区域**：

   - ```
     canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3]
     ```

     - 这部分是通过数组切片选择 `canvas` 上的一个区域。`coord` 表示起始坐标（y坐标和x坐标），`patch_size` 表示从这个坐标点开始，需要替换或覆盖的区域大小（高度和宽度）
     - `:3` 指选择所有色彩通道

2. **定义 `patch` 图像的使用区域**：

   - ```python
     patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
     ```

     - 这部分切片操作选取 `patch` 的一个子区域
     - 切片 `[:canvas_crop_shape[0], :canvas_crop_shape[1]]` 确保了从 `patch` 中选取的区域不会超出 `canvas` 的目标区域尺寸，避免尺寸不匹配的错误