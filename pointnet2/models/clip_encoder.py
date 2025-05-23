import torch
import clip
from torchvision.transforms.functional import to_pil_image


class CLIPEncoder:
    def __init__(self, class_names, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.category_names = ["an image of a " + class_name for class_name in class_names]
        self.out_dim = 512

    
    def get_img_features(self, batch_images):
        with torch.no_grad():
            pil_images = [to_pil_image(image) for image in batch_images]
            processed_batch_images = [self.preprocess(pil_image) for pil_image in pil_images]
            batch_images = torch.stack(processed_batch_images).to(self.device)
            return self.model.encode_image(batch_images).float()
    
    def get_image_features(self, class_index):
        if class_index.ndim > 1:
            return self.get_img_features(class_index)
           
        processed_text = []
        for index in class_index:
            processed_text.append(clip.tokenize(self.category_names[index]))
        processed_text = torch.cat(processed_text, dim=0).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(processed_text).float()
