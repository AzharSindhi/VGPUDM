# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms as pth_transforms
from . import vision_transformer as vits

class DinoEncoder:
    def __init__(self, image_size=480, arch = 'vit_small', patch_size = 8, aggregate="none", device="cuda") -> None:
        # convert args to init parameters


        self.device = torch.device(device)
        self.patch_size = patch_size
        # build model
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        self.normalize_transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            assert False, "There is no reference weights available for this model => We use random weights."

        self.aggregate = aggregate
        self.model = model
        self.out_dim = 384 #model.embed_dim
        self.to_pil_test = pth_transforms.ToPILImage()

    def get_image_features(self, img):

        # transform = pth_transforms.Compose([
        #     pth_transforms.Resize(image_size),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        
        img = self.normalize_transform(img)
        # img_test = self.to_pil_test(img[0])
        # img_test.save("test.png")
        # image has already batch dimension
        # make the image divisible by the patch size
        w, h = img.shape[2] - img.shape[2] % self.patch_size, img.shape[3] - img.shape[3] % self.patch_size
        img = img[:, :, :w, :h]
        
        attentions = self.model.get_intermediate_layers(img.to(self.device), n=1)
        class_token = None
        if self.aggregate == "class":
            # get the class token embedding
            class_token = attentions[0][:, 0, :]
        elif self.aggregate == "avgpool":
            # average pooling across second dim
            output = torch.cat([x[:, 0] for x in attentions], dim=-1)
            output = torch.cat((output.unsqueeze(-1), torch.mean(attentions[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            class_token = output.reshape(output.shape[0], -1)
        elif self.aggregate == "none":
            class_token = torch.cat(attentions, dim=-1)
        else:
            raise ValueError("Invalid aggregate type")

        return class_token