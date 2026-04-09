import torch
import torch.nn as nn


class FusionNet(nn.Module):
    """
    针对任意多模态/单模态情感识别的分类网络。
    支持纯 MLP 架构 或 时序(GRU) + MLP 架构。
    自动适配 Video, EEG, ECG 模态，支持任意数量输入融合。
    """

    def __init__(self, feature_types=['deep_feature'], num_classes=5, use_gru=True, dropout_rate=0.0):
        """
        参数:
            feature_types: 特征类型列表 (如 ['deep_feature', 'eeg_de_feats'])。单特征可直接传字符串。
            num_classes: 分类类别数 (如 2, 3, 5, 6)
            use_gru: 是否使用单向 GRU 提取时序特征。如果为 False，则使用纯 MLP(Flatten)。
            dropout_rate: 防止过拟合的丢弃率
        """
        super(FusionNet, self).__init__()

        # 兼容单特征传入字符串的情况
        if isinstance(feature_types, str):
            feature_types = [feature_types]

        self.feature_types = feature_types
        self.num_classes = num_classes
        self.use_gru = use_gru

        # 1. 特征字典: 映射 feature_type -> (input_dim, default_frames)
        self.feature_info_map = {
            # Video Modality
            'au_openface': (35, 16),
            'hp_openface': (6, 16),
            'gz_openface': (2, 16),
            'deep_feature': (512, 16),
            # EEG Modality
            'eeg_de_feats': (40, 1),
            'eeg_psd_feats': (40, 1),
            'eeg_hfd_feats': (8, 1),
            'eeg_sampen_feats': (8, 1),
            # ECG Modality
            'ecg_time_feats': (5, 1),
            'ecg_freq_feats': (3, 1),
            'ecg_hfd_feats': (1, 1),
            'ecg_sampen_feats': (1, 1),
        }

        # 隐藏层维度设定
        hidden_dim = 128

        # 存放所有模态的特征提取器
        self.extractors = nn.ModuleList()
        classifier_in_dim = 0  # 动态计算最终融合后的特征维度

        # 2. 为每个传入的模态动态构建特征提取器
        for f_type in self.feature_types:
            # ======= 新增：支持多特征列表拼接计算总维度 =======
            if isinstance(f_type, list):
                input_dim = 0
                default_frames = None
                for sub_f in f_type:
                    if sub_f not in self.feature_info_map:
                        raise ValueError(f"未知的特征类型 {sub_f}。请从 {list(self.feature_info_map.keys())} 中选择。")
                    sub_dim, sub_frames = self.feature_info_map[sub_f]
                    input_dim += sub_dim  # 累加各个特征的维度
                    default_frames = sub_frames  # 假设拼接的特征帧数一致
            else:
                if f_type not in self.feature_info_map:
                    raise ValueError(f"未知的特征类型 {f_type}。请从 {list(self.feature_info_map.keys())} 中选择。")
                input_dim, default_frames = self.feature_info_map[f_type]

            if self.use_gru:
                # 架构 A: GRU
                extractor = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,  # 单向
                    dropout=0  # 当 num_layers=1 时，PyTorch 的 GRU 不接受 dropout>0
                )
                out_dim = hidden_dim
            else:
                # 架构 B: Flatten
                extractor = nn.Flatten(start_dim=1)
                # 修复: Flatten 后的维度严格等于 input_dim * 帧数，保证任意模态都不会发生形状不匹配
                out_dim = input_dim * default_frames

            self.extractors.append(extractor)
            classifier_in_dim += out_dim

        # 3. 分类器头 (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.num_classes)
        )

    def forward(self, *inputs):
        """
        前向传播
        输入 inputs: 可变长张量元组 (x1, x2, ...)，顺序须与初始化的 feature_types 一致
        """
        if len(inputs) != len(self.feature_types):
            raise ValueError(f"前向传播传入了 {len(inputs)} 个张量，但网络初始化了 {len(self.feature_types)} 个模态！")

        extracted_features = []

        # 遍历每个模态提取特征
        for i, (x, f_type) in enumerate(zip(inputs, self.feature_types)):
            extractor = self.extractors[i]

            if self.use_gru:
                # GRU 输出: out 形状为 (batch_size, frames, hidden_dim)
                out, _ = extractor(x)
                # 时序池化: 在时间维度上求平均
                feat = torch.mean(out, dim=1)
            else:
                # Flatten 输出: (batch_size, frames * input_dim)
                feat = extractor(x)

            extracted_features.append(feat)

        # 多模态特征拼接 (如果是单模态，这步相当于直接传递)
        if len(extracted_features) > 1:
            fused_features = torch.cat(extracted_features, dim=1)
        else:
            fused_features = extracted_features[0]

        # 通过 MLP 进行分类
        logits = self.classifier(fused_features)

        return {"y_pred": logits}


def test_networks():
    from thop import profile

    # --- 测试用例定义：覆盖单模态、双模态、三模态 ---
    test_cases = [
        # {"name": "Single: Video (GRU)", "feats": ["deep_feature"], "gru": True, "bs": 2, "shapes": [(2, 16, 512)]},
        # {"name": "Single: EEG (MLP)", "feats": ["eeg_de_feats"], "gru": False, "bs": 2, "shapes": [(2, 1, 40)]},
        # {"name": "Fusion: Vid+EEG (GRU)", "feats": ["deep_feature", "eeg_de_feats"], "gru": True, "bs": 2,
        #  "shapes": [(2, 16, 512), (2, 1, 40)]},
        # {"name": "Fusion: Vid+EEG+ECG (MLP)", "feats": ["deep_feature", "eeg_sampen_feats", "ecg_time_feats"],
        #  "gru": False, "bs": 2, "shapes": [(2, 16, 512), (2, 1, 8), (2, 1, 5)]},
        {"name": "Fusion: Vid+EEG+ECG (GRU)", "feats": ["hp_openface", "eeg_hfd_feats", "ecg_hfd_feats"],
         "gru": False, "bs": 2, "shapes": [(2, 16, 6), (2, 1, 8), (2, 1, 1)]},
    ]

    print(f"{'测试用例':<28} | {'参数量 (M)':<12} | {'MACs (M)':<10} | {'输出形状'}")
    print("-" * 75)

    for case in test_cases:
        model = FusionNet(feature_types=case["feats"], num_classes=5, use_gru=case["gru"])
        model.eval()

        # 生成对应数量的 dummy 输入
        dummy_inputs = [torch.randn(*shape) for shape in case["shapes"]]

        # 测试前向传播 (使用 * 解包传入)
        output = model(*dummy_inputs)
        out_shape = str(list(output["y_pred"].shape))

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6

        # 计算复杂度 (基于单样本计算)
        single_inputs = tuple(torch.randn(1, *shape[1:]) for shape in case["shapes"])
        macs, _ = profile(model, inputs=single_inputs, verbose=False)
        macs_m = macs / 1e6

        # 打印格式化输出
        macs_str = f"{macs_m:<10.4f}" if isinstance(macs_m, float) else f"{macs_m:<10}"
        print(f"{case['name']:<28} | {total_params:<12.4f} | {macs_str} | {out_shape}")

    print("-" * 75)


if __name__ == "__main__":
    test_networks()
