from kornia import bgr_to_grayscale
import torch
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
yolov5s = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
load_model = torch.jit.load('model.pth').cuda().eval()

class HumanDetection:
    def __init__(self, model,  offset = 100, return_one = True) -> None:
        self.offset = offset
        self.model = model
        self.offset = offset
        self.return_one = return_one
        
    def obtain_coordinate(self, imgs) -> list or tuple:
        self.results = self.model(imgs)
        human_xyxy = []
        if self.return_one:
            max_conf = 0
            max_coor = None
            for clses in self.results.xyxy[0]:
                *xyxy, conf, name = clses
                if name.item() == 0 and max_conf < conf:  # class number 0
                    max_coor = self.rect_area(xyxy)
            return max_coor
        else:
            human_xyxy = []
            for clses in self.results.xyxy[0]:
                *xyxy, conf, name = clses
                if name.item() == 0: # class number 0
                    human_xyxy.append(self.rect_area(xyxy))
            return human_xyxy
    
    def rect_area(self, xyxy):
        xmin, ymin, xmax, ymax = xyxy
        xmin = max(0,xmin.item()-self.offset)
        ymin = max(0,ymin.item()-self.offset)
        xmax = max(0,xmax.item()+self.offset)
        ymax = max(0,ymax.item()+self.offset)
        return xmin, ymin, xmax, ymax

    def maskxyxy(self, imgs) -> torch.Tensor:
        human_xyxy = self.obtain_coordinate(imgs)
        x,y = None, None
        if hasattr(imgs, 'shape'):
            x,y = imgs[0].shape 
        else:
            x,y = imgs.size
        mask = torch.zeros((y,x), device='cuda')
        if self.return_one:
            xmin, ymin, xmax, ymax = human_xyxy
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        else:
            for xyxy in human_xyxy:
                xmin, ymin, xmax, ymax = xyxy
                mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        return mask
            

class ImageMatting:
    def __init__(self, load_model, trained_model_path,) -> None:
        self.model = load_model
        self.model.load_state_dict(torch.load(trained_model_path), strict= True)
        
    def __call__(self, bgrs, srcs):
        if srcs.size(2) <= 2048 and srcs.size(3) <= 2048:
            self.model.backbone_scale = 1/4
            self.model.refine_sample_pixels = 80_000
        else:
            self.model.backbone_scale = 1/8
            self.model.refine_sample_pixels = 320_000
            
        pha, fgr = self.model(srcs, bgrs)[:2]
        # com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
        return pha, fgr

        
if __name__ == '__main__':
    src_path = 'test_src.png'
    bgr_path = 'test_bgr.png'
    img_src = Image.open(src_path).convert('RGB')
    img_bgr = Image.open(bgr_path).convert('RGB')
    tensor_src = to_tensor(img_src).cuda().unsqueeze(0)
    tensor_bgr = to_tensor(img_bgr).cuda().unsqueeze(0)
    img_mat = ImageMatting(load_model = load_model, 
                            trained_model_path = './checkpoint/mattingrefine-resnet50-videomatte240k/epoch-0.pth')
    
    pha, fgr = img_mat(bgrs = tensor_bgr,srcs = tensor_src)
    human_detect = HumanDetection(model = yolov5s,return_one = True)
    mask = human_detect.maskxyxy(imgs = img_src)
    pha = pha * mask
    com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
    plt.imshow(to_pil_image(fgr[0].cpu()))
    plt.savefig('forground.png')
    plt.imshow(to_pil_image(com[0].cpu()))
    plt.savefig('result.png')
    
    
    
    
    
