import torch
import torch.nn as nn
from .utils import Dropout
from .convolutions import ConvBnAct
from .activations import LeakyReLU
import torch.nn.functional as F
import configparser


class ClassificationHead(nn.Sequential):
    """
        Classification Head for Neural Networks.
        This module represents a classification head typically used at the end of a neural network. It consists of a
        linear layer, a ReLU activation, dropout for regularization, and a final linear layer that maps to the number
        of classes. This head is designed to be attached to the feature-extracting layers of a network to perform
        classification tasks.

        Args:
            input_size (int): The size of the input features.
            num_classes (int, optional): The number of classes for classification. Defaults to 2.

        The sequence of operations is as follows: Linear -> ReLU -> Dropout -> Linear.
    """
    def __init__(self, input_size, num_classes=2):
        super(ClassificationHead, self).__init__(
            #nn.Linear(input_size, 512),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            Dropout(p=0.4),
            #nn.Linear(512, num_classes)
            nn.Linear(256, num_classes)
        )


###################################################################################################################################
# YOLO Object Detection head, inpired by https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py
"""ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
"""
ANCHORS = [ # obtained by normalizing over 416 the classic anchors
    (0.02403846153846154, 0.03125),
    (0.038461538461538464, 0.07211538461538461),
    (0.07932692307692307, 0.055288461538461536),
    (0.07211538461538461, 0.1466346153846154),
    (0.14903846153846154, 0.10817307692307693),
    (0.14182692307692307, 0.2860576923076923),
    (0.27884615384615385, 0.21634615384615385),
    (0.375, 0.47596153846153844),
    (0.8966346153846154, 0.7836538461538461)
    ] #"""


class YoloLayer(nn.Module):
    """
        YOLO Layer for handling detection at different scales.
    """
    def __init__(self, scale, stride, num_classes, num_anchors_per_scale=3):
        super(YoloLayer, self).__init__()
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            idx = None

        self.num_classes = num_classes
        # num_attrib is 4 for the bbox coordinates, 1 for prediction score, and then classification scores for each class
        self.num_attrib = 4 + 1 + self.num_classes
        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride
        self.num_anchors_per_scale = num_anchors_per_scale

    def forward(self, x):
        #print("Shape of input to YOLO layer:", x.shape)
        num_batch = x.size(0)
        num_grid = x.size(2) # squared grid.
        #print("GRID SIZE:", num_grid, "x", num_grid)
        if torch.isnan(x).any():
            print("NaNs found in raw output during training (head)")

        #if self.training:
        #    output_raw = x.view(num_batch,
        #                        self.num_anchors_per_scale,
        #                        self.num_attrib,
        #                        num_grid,
        #                        num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, self.num_attrib)
        #    return output_raw
        #else:
        # Check for NaNs in the raw output
        
        prediction_raw = x.view(num_batch,
                                self.num_anchors_per_scale,
                                self.num_attrib,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous()

        self.anchors = self.anchors.to(x.device).float()
        # Calculate offsets for each grid
        grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
        grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
        grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
        anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
        anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

        # Get outputs
        x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride / (416 + 1e-9) # Center x
        y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride / (416 + 1e-9) # Center y
        w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
        h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
        bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
        conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
        cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, self.num_classes)  # Cls pred one-hot.

        output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
        return output


class DetectionBlock(nn.Module):
    """
    The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """
    def __init__(self, in_channels, out_channels, scale, stride, num_classes, num_anchors_per_scale=3):
        super(DetectionBlock, self).__init__()
        assert out_channels % 2 == 0  #assert out_channels is an even number
        half_out_channels = out_channels // 2
        self.conv1 = ConvBnAct(in_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv2 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv3 = ConvBnAct(out_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv4 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv5 = ConvBnAct(out_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv6 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv7 = nn.Conv2d(out_channels, (4+1+num_classes) * num_anchors_per_scale, kernel_size=1, bias=True)
        self.yolo = YoloLayer(scale, stride, num_classes=num_classes)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        self.branch = self.conv5(tmp)
        tmp = self.conv6(self.branch)
        tmp = self.conv7(tmp)
        out = self.yolo(tmp)

        return out
    

class DetectionBlock_small(nn.Module):
    """
    A lighter version of DetectionBlock. 
    """
    def __init__(self, in_channels, out_channels, scale, stride, num_classes, num_anchors_per_scale=3):
        super(DetectionBlock_small, self).__init__()
        assert out_channels % 2 == 0  #assert out_channels is an even number
        half_out_channels = out_channels // 2
        self.conv1 = ConvBnAct(in_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv2 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv3 = ConvBnAct(out_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv4 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv7 = nn.Conv2d(out_channels, (4+1+num_classes) * num_anchors_per_scale, kernel_size=1, bias=True)
        self.yolo = YoloLayer(scale, stride, num_classes=num_classes)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        self.branch = self.conv3(tmp)
        tmp = self.conv4(self.branch)
        tmp = self.conv7(tmp)
        out = self.yolo(tmp)
        return out


class DetectionHeadYOLOv3(nn.Module): # Formerly YoloNetTail
    """
    The tail side of the YoloNet.
    In YOLOv3, it will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Takes in input 3 tensors like these:
        in1 = torch.randn((2, 1024, IMAGE_SIZE//4, IMAGE_SIZE//4))
        in2 = torch.randn((2, 512, IMAGE_SIZE//2, IMAGE_SIZE//2))
        in3 = torch.randn((2, 256, IMAGE_SIZE, IMAGE_SIZE))
    """
    def __init__(self, num_classes=2):
        super().__init__()

        config = configparser.ConfigParser()
        config.read('config.ini')
        self.input_size = config.getint(section='DetectionHeadYOLOv3', option='image_size')
        num_anchors_per_scale = config.getint(section='DetectionHeadYOLOv3', option='num_anchors_per_scale')

        self.num_classes = num_classes
        self.detect1 = DetectionBlock(1024, 1024, 'l', 32, num_classes=self.num_classes, num_anchors_per_scale=num_anchors_per_scale)
        self.conv1 = ConvBnAct(512, 256, 1, activation=LeakyReLU)
        self.detect2 = DetectionBlock(768, 512, 'm', 16, num_classes=self.num_classes, num_anchors_per_scale=num_anchors_per_scale)
        self.conv2 = ConvBnAct(256, 128, 1, activation=LeakyReLU)
        self.detect3 = DetectionBlock(384, 256, 's', 8, num_classes=self.num_classes, num_anchors_per_scale=num_anchors_per_scale)

    def forward(self, inputs):
        x1 = inputs[0]; x2 = inputs[1]; x3 = inputs[2]
        #print("INPUTS: x1: ", x1.shape, "- x2: ", x2.shape, "- x3: ", x3.shape)

        # Resizing input tensors to fit the canon YOLOv3
        if x1.shape[2] > self.input_size//32: # for 32 stride
            x1 = F.adaptive_avg_pool2d(x1, (self.input_size//32, self.input_size//32))
        else:
            x1 = F.interpolate(x1, size=(self.input_size//32, self.input_size//32), mode='bilinear', align_corners=False) # for 32 stride

        if x2.shape[2] > self.input_size//16: # for 16 stride
            x2 = F.adaptive_avg_pool2d(x2, (self.input_size//16, self.input_size//16))
        else:
            x2 = F.interpolate(x2, size=(self.input_size//16, self.input_size//16), mode='bilinear', align_corners=False) # for 32 stride

        if x3.shape[2] > self.input_size//8: # for 8 stride
            x3 = F.adaptive_avg_pool2d(x3, (self.input_size//8, self.input_size//8))
        else:
            x3 = F.interpolate(x3, size=(self.input_size//8, self.input_size//8), mode='bilinear', align_corners=False) # for 32 stride

        #print("AFTER DOWNSCALE: x1: ", x1.shape, "- x2: ", x2.shape, "- x3: ", x3.shape)

        out1 = self.detect1(x1)
        branch1 = self.detect1.branch
        tmp = self.conv1(branch1)
        
        # Resize tmp to match the shape of x2, then concatenate
        tmp = F.interpolate(tmp, size=x2.shape[2:])
        tmp = torch.cat((tmp, x2), 1)
        out2 = self.detect2(tmp)
        branch2 = self.detect2.branch
        tmp = self.conv2(branch2)
        
        # Resize tmp to match the shape of x3, then concatenate
        tmp = F.interpolate(tmp, size=x3.shape[2:])
        tmp = torch.cat((tmp, x3), 1)
        out3 = self.detect3(tmp)

        #return out1, out2, out3

        out = torch.cat((out1, out2, out3), 1)
        
        return out


###################################################################################################################################
# YOLO Object Detection head, but smaller

"""
#ANCHORS_SmallObjects = [(10, 25), (15, 15), (25, 10), (25, 50), (35, 35), (50, 25)]
ANCHORS_SmallObjects = [(19, 14), (22, 29), (27, 44), (30, 17), (45, 21), (73, 25)]
    
"""
'''
ANCHORS_SmallObjects = [# obtained by normalizing over 512
    (0.01953125, 0.04882812),
    (0.02929688, 0.02929688),
    (0.04882812, 0.01953125),
    (0.04882812, 0.09765625),
    (0.06835938, 0.06835938),
    (0.09765625, 0.04882812)
    ]
'''

#Anchor Box 1: 18.81 x 14.03 with aspect ratio 1.41
#Anchor Box 2: 21.75 x 28.86 with aspect ratio 0.77
#Anchor Box 4: 29.78 x 16.55 with aspect ratio 1.89
#Anchor Box 3: 27.00 x 44.23 with aspect ratio 0.62
#Anchor Box 5: 45.17 x 21.15 with aspect ratio 2.25
#Anchor Box 6: 72.67 x 24.50 with aspect ratio 3.15

ANCHORS_SmallObjects = [# obtained by normalizing over 512
    (0.03673828, 0.02740234),
    (0.04248047, 0.05636719),
    (0.05816406, 0.03232422),
    (0.05273438, 0.08638672),
    (0.08822266, 0.04130859),
    (0.14193359, 0.04785156)
]

class YoloLayer_SmallObjects(nn.Module):
    """
        YOLO Layer for handling detection at different scales.
    """
    def __init__(self, scale, stride, num_classes=1, num_anchors_per_scale=3):
        super(YoloLayer_SmallObjects, self).__init__()
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        else:
            idx = None

        self.num_classes = num_classes
        # num_attrib is 4 for the bbox coordinates, 1 for prediction score, and then classification scores for each class
        self.num_attrib = 4 + 1 + self.num_classes
        self.anchors = torch.tensor([ANCHORS_SmallObjects[i] for i in idx])
        self.stride = stride
        self.num_anchors_per_scale = num_anchors_per_scale

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)
        prediction_raw = x.view(
            num_batch,
            self.num_anchors_per_scale,
            self.num_attrib,
            num_grid,
            num_grid
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        if self.training:
            # During training, output raw predictions
            #print("Model in training mode:")
            return self.transform_predictions(prediction_raw, num_grid) # prediction_raw.view(num_batch, -1, self.num_attrib), 
        else:
            # During inference, apply transformations to get bounding box coordinates. Not implemented with the lightning module.
            #print("Model in inference mode:")
            return self.transform_predictions(prediction_raw, num_grid)

    def transform_predictions(self, prediction_raw, num_grid):
        # This method is used during inference
        num_batch = prediction_raw.size(0)
        device = prediction_raw.device

        # Prepare anchors and grid
        self.anchors = self.anchors.to(device).float()
        grid_x, grid_y = self._create_grids(num_grid, device)
        anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
        anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

        # Transform raw predictions to bounding box coordinates
        x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride / 512  # Adjust as per your scaling
        y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride / 512
        w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w
        h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h

        bbox_pred = torch.stack(
            (x_center_pred, y_center_pred, w_pred, h_pred), dim=4
        ).view((num_batch, -1, 4))  # cxcywh
        conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)
        cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, self.num_classes)

        output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
        return output

    def _create_grids(self, num_grid, device):
        grid_tensor = torch.arange(num_grid, dtype=torch.float, device=device)
        grid_x = grid_tensor.repeat(num_grid, 1).view([1, 1, num_grid, num_grid])
        grid_y = grid_tensor.repeat(num_grid, 1).t().view([1, 1, num_grid, num_grid])
        return grid_x, grid_y

    '''def forward(self, x):
        #print("Shape of input to YOLO layer:", x.shape)
        num_batch = x.size(0)
        num_grid = x.size(2) # squared grid.
        #print("GRID SIZE:", num_grid, "x", num_grid)

        if torch.isnan(x).any():
            print("NaNs found in raw output during training (head)")
        
        prediction_raw = x.view(num_batch,
                                self.num_anchors_per_scale,
                                self.num_attrib,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous()

        self.anchors = self.anchors.to(x.device).float()
        # Calculate offsets for each grid
        grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
        grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
        grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
        anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
        anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

        # Get outputs 
        ## Old implementation with bbox coordinates being calculted from offsets in the head. Now moved to loss and post-processing.
        #x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride / 512 # Center x
        #y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride / 512 # Center y
        #w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
        #h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
        #bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
        x_center_raw = prediction_raw[..., 0]
        y_center_raw = prediction_raw[..., 1]
        w_raw = prediction_raw[..., 2]
        h_raw = prediction_raw[..., 3]
        bbox_pred = torch.stack((x_center_raw, y_center_raw, w_raw, h_raw), dim=4).view((num_batch, -1, 4))  # Raw predictions
        conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
        cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, self.num_classes)  # Cls pred one-hot.

        output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
        return output'''


class DetectionBlock_SmallObjects(nn.Module):
    """
    The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """
    def __init__(self, in_channels, out_channels, scale, stride, num_classes, num_anchors_per_scale=3):
        super(DetectionBlock_SmallObjects, self).__init__()
        assert out_channels % 2 == 0  #assert out_channels is an even number
        half_out_channels = out_channels // 2
        self.conv1 = ConvBnAct(in_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv2 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv3 = ConvBnAct(out_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv4 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv5 = ConvBnAct(out_channels, half_out_channels, kernel_size=1, activation=LeakyReLU)
        self.conv6 = ConvBnAct(half_out_channels, out_channels, kernel_size=3, activation=LeakyReLU)
        self.conv7 = nn.Conv2d(out_channels, (4+1+num_classes) * num_anchors_per_scale, kernel_size=1, bias=True)
        self.yolo = YoloLayer_SmallObjects(scale, stride, num_classes=num_classes)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        self.branch = self.conv5(tmp)
        tmp = self.conv6(self.branch)
        tmp = self.conv7(tmp)
        out = self.yolo(tmp)

        return out
    

class DetectionHeadYOLOv3_SmallObjects(nn.Module):
    """
    The tail side of the YoloNet.
    In YOLOv3, it will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Takes in input 3 tensors like these:
        in1 = torch.randn((2, 1024, IMAGE_SIZE//4, IMAGE_SIZE//4))
        in2 = torch.randn((2, 512, IMAGE_SIZE//2, IMAGE_SIZE//2))
        in3 = torch.randn((2, 256, IMAGE_SIZE, IMAGE_SIZE))
    """
    def __init__(self, num_classes=1):
        super().__init__()

        config = configparser.ConfigParser()
        config.read('config.ini')
        self.input_size = config.getint(section='DetectionHeadYOLOv3_SmallObjects', option='image_size')
        num_anchors_per_scale = config.getint(section='DetectionHeadYOLOv3_SmallObjects', option='num_anchors_per_scale')

        self.num_classes = num_classes
        self.detect1 = DetectionBlock_SmallObjects(512, 512, 'm', 16, num_classes=self.num_classes, num_anchors_per_scale=num_anchors_per_scale)
        self.conv1 = ConvBnAct(256, 128, 1, activation=LeakyReLU)
        self.detect2 = DetectionBlock_SmallObjects(384, 256, 's', 8, num_classes=self.num_classes, num_anchors_per_scale=num_anchors_per_scale)

    def forward(self, inputs):
        x1 = inputs[0]; x2 = inputs[1]
        #print("INPUTS: x1: ", x1.shape, "- x2: ", x2.shape)

        if x1.shape[2] > self.input_size//16: # for 16 stride
            x1 = F.adaptive_avg_pool2d(x1, (self.input_size//16, self.input_size//16))
        else:
            x1 = F.interpolate(x1, size=(self.input_size//16, self.input_size//16), mode='bilinear', align_corners=False) # for 32 stride

        if x2.shape[2] > self.input_size//8: # for 8 stride
            x2 = F.adaptive_avg_pool2d(x2, (self.input_size//8, self.input_size//8))
        else:
            x2 = F.interpolate(x2, size=(self.input_size//8, self.input_size//8), mode='bilinear', align_corners=False) # for 32 stride

        #print("AFTER DOWNSCALE: x1: ", x1.shape, "- x2: ", x2.shape)

        out1 = self.detect1(x1)
        branch1 = self.detect1.branch
        tmp = self.conv1(branch1)
        
        # Resize tmp to match the shape of x2, then concatenate
        tmp = F.interpolate(tmp, size=x2.shape[2:])
        tmp = torch.cat((tmp, x2), 1)
        out2 = self.detect2(tmp)

        out = torch.cat((out1, out2), 1)

        #print("out1: ", out1.shape, "- out2: ", out2.shape, "- out: ", out.shape)
        
        return out
    