# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.act='lrelu'
    # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0.01
        # default lr
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 0
        self.min_lr_ratio = 0
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = 'datasets/Hailo_EVS2022_Demo_Training_Data'
        self.train_ann = 'instances_train2017.json'
        self.val_ann = 'instances_val2017.json'
        self.num_classes = 1

    def get_model(self):
        import torch.nn as nn
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            if self.backbone is not None:
                backbone = torch.load(self.backbone)
            else:
                backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(80, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        import ast
                        v = ast.literal_eval(v)
                setattr(self, k, v)
            else:
                setattr(self, k, v)