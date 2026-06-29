import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerModel


class SegFormerMultiTask(nn.Module):

    def __init__(
            self,
            num_classes=3,
            seg_classes=1,
            pretrained=True):

        super().__init__()

        model_name = "nvidia/mit-b2"

        self.encoder = SegformerModel.from_pretrained(
            model_name
        )

        #################################################
        # Freeze all encoder layers
        #################################################

        for param in self.encoder.parameters():
            param.requires_grad = False

        #################################################
        # Unfreeze last 2 stages
        #################################################

        for param in self.encoder.encoder.block[2].parameters():
            param.requires_grad = True

        for param in self.encoder.encoder.block[3].parameters():
            param.requires_grad = True

        #################################################
        # Segmentation Decoder
        #################################################

        self.conv1 = nn.Conv2d(64, 128, 1)
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.conv3 = nn.Conv2d(320, 128, 1)
        self.conv4 = nn.Conv2d(512, 128, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(
            128,
            seg_classes,
            kernel_size=1
        )

        #################################################
        # Classification Head
        #################################################

        self.cls_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),

        nn.Linear(128, 3)
        )

    def forward(self, x):

        input_size = x.shape[2:]

        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True
        )

        features = outputs.hidden_states

        f1 = features[0]     # 64
        f2 = features[1]     # 128
        f3 = features[2]     # 320
        f4 = features[3]     # 512

        #################################################
        # Classification Branch
        #################################################

        cls = F.adaptive_avg_pool2d(
            f4,
            1
        ).flatten(1)

        logits = self.cls_head(cls)

        #################################################
        # Segmentation Branch
        #################################################

        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)

        f2 = F.interpolate(
            f2,
            size=f1.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        f3 = F.interpolate(
            f3,
            size=f1.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        f4 = F.interpolate(
            f4,
            size=f1.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x_seg = torch.cat(
            [f1, f2, f3, f4],
            dim=1
        )

        x_seg = self.fuse(x_seg)

        mask = self.seg_head(x_seg)

        mask = F.interpolate(
            mask,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )

        return mask, logits