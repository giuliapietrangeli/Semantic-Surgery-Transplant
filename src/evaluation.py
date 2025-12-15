import torch
import cv2
import numpy as np
import lpips
import gc
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from torchvision.models import resnet50, ResNet50_Weights
from transformers import (
    CLIPProcessor, CLIPModel,
    ViTForImageClassification, ViTImageProcessor,
    BlipProcessor, BlipForConditionalGeneration,
    OwlViTProcessor, OwlViTForObjectDetection
)

class Evaluator:
    def __init__(self, device="cpu"):
        print(f"Initializing Evaluator (Lazy Loading Mode) on {device}...")
        self.device = device
        
        self.resnet = None
        self.clip_model = None
        self.owl_model = None
        self.lpips = None
        self.vit = None
        self.blip = None
        
        self.weights_rn = ResNet50_Weights.DEFAULT
        self.rn_preprocess = self.weights_rn.transforms()
        self.lpips_trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.SWAP_INDICES = {
            "bear": 294, "brown bear": 294, "dog": 263, "goldfish": 1, "fish": 1,
            "sportscar": 817, "car": 817, "sports car": 817,
            "mug": 504, "apple": 948, "tiger": 292, "cat": 285, "tabby": 285,
            "shark": 2, "white shark": 2, "firetruck": 555, "fire truck": 555,
            "beer": 440, "beer bottle": 440, "daisy": 985, "flower": 985,
            "sofa": 831, "studio couch": 831, "yawl": 914, "boat": 914, "corgi": 263, "brown bear": 294
        }
        
        self.TARGET_INDICES = {
            "goldfish": 1, "bear": 294, "corgi": 263,
            "sportscar": 817, "sofa": 831, "yawl": 914
        }
        
        self.gradients = None
        self.activations = None
    
    def _ensure_resnet(self):
        if self.resnet is None:
            print("   Loading ResNet-50...")
            self.resnet = resnet50(weights=self.weights_rn).to(self.device).eval()
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            def forward_hook(module, input, output):
                self.activations = output
            self.resnet.layer4[2].conv3.register_forward_hook(forward_hook)
            self.resnet.layer4[2].conv3.register_full_backward_hook(backward_hook)

    def _ensure_clip(self):
        if self.clip_model is None:
            print("   Loading CLIP...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _ensure_owl(self):
        if self.owl_model is None:
            print("   Loading Owl-ViT...")
            self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
            self.owl_proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    def _ensure_lpips(self):
        if self.lpips is None:
            print("   Loading LPIPS...")
            self.lpips = lpips.LPIPS(net='alex').to(self.device)

    def _ensure_vit(self):
        if self.vit is None:
            print("   Loading ViT...")
            name = 'google/vit-base-patch16-224'
            self.vit = ViTForImageClassification.from_pretrained(name, output_attentions=True).to(self.device).eval()
            self.vit_proc = ViTImageProcessor.from_pretrained(name)

    def _ensure_blip(self):
        if self.blip is None:
            print("   Loading BLIP...")
            self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def free_memory(self):
        self.resnet = None
        self.clip_model = None
        self.owl_model = None
        self.lpips = None
        self.vit = None
        self.blip = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("   Memory Cleared.")
    
    def calculate_structural_metrics(self, path1, path2):
        return self.get_structural_metrics(path1, path2)

    def get_structural_metrics(self, p1, p2):
        self._ensure_lpips()
        if not os.path.exists(p1) or not os.path.exists(p2): return 0,0,0
        
        i1 = cv2.imread(p1); i2 = cv2.imread(p2)
        if i1.shape != i2.shape: i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        s_score, _ = ssim(g1, g2, full=True)
        
        m_score = mean_squared_error(g1, g2)
        
        l_score = 0
        if self.lpips:
            pil1 = Image.open(p1).convert('RGB'); pil2 = Image.open(p2).convert('RGB')
            t1 = self.lpips_trans(pil1).unsqueeze(0).to(self.device)
            t2 = self.lpips_trans(pil2).unsqueeze(0).to(self.device)
            with torch.no_grad(): l_score = self.lpips(t1, t2).item()
        
        return s_score, m_score, l_score

    def get_ssim_score(self, img1, img2):
        i1 = np.array(img1.convert('L'))
        i2 = np.array(img2.convert('L'))
        if i1.shape != i2.shape: i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        score, _ = ssim(i1, i2, full=True)
        return score

    def get_clip_score_single(self, img, text):
        return self.get_clip_score(img, text)

    def get_clip_score(self, img, text):
        self._ensure_clip()
        inputs = self.clip_proc(text=[text[:77]], images=img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(): outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()

    def get_clip_similarity(self, img_path, text_prompts):
        self._ensure_clip()
        image = Image.open(img_path).convert("RGB")
        inputs = self.clip_proc(text=text_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(): outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    def get_clip_logits(self, img, text_list):
        self._ensure_clip()
        inputs = self.clip_proc(text=[t[:77] for t in text_list], images=img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(): outputs = self.clip_model(**inputs)
        return outputs.logits_per_image[0].cpu().numpy()

    def get_resnet_conf(self, pil_image, target_name):
        self._ensure_resnet()
        if pil_image is None: return 0.0
        target_idx = self.SWAP_INDICES.get(target_name.lower())
        if target_idx is None:
             for k, v in self.SWAP_INDICES.items():
                if k in target_name.lower(): target_idx = v; break
        if target_idx is None: return 0.0
        try:
            t = self.rn_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad(): probs = self.resnet(t).softmax(1)
            return probs[0, target_idx].item()
        except: return 0.0
    
    def get_target_score(self, pil_image, target_idx):
        self._ensure_resnet()
        if pil_image is None: return 0.0
        t = self.rn_preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad(): probs = self.resnet(t).softmax(1)
        return probs[0, target_idx].item()

    def get_top2_verdict(self, pil_image):
        self._ensure_resnet()
        if pil_image is None: return []
        t = self.rn_preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.resnet(t).softmax(1)
        scores, ids = torch.topk(probs, 2)
        results = []
        for i in range(2):
            results.append({
                "name": self.weights_rn.meta["categories"][ids[0][i].item()],
                "score": scores[0][i].item(),
                "id": ids[0][i].item()
            })
        return results

    def compute_gradcam(self, pil_image, target_class_index):
        self._ensure_resnet()
        img_tensor = self.rn_preprocess(pil_image).unsqueeze(0).to(self.device)
        self.resnet.zero_grad()
        output = self.resnet(img_tensor)
        score = output[0, target_class_index]
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations_data = self.activations[0]
        for i in range(activations_data.shape[0]):
            activations_data[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations_data, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
            
        img_np = np.array(pil_image)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        return np.uint8(heatmap_colored * 0.4 + img_np * 0.6)

    def get_gender_prob(self, img):
        logits = self.get_clip_logits(img, ["a photo of a male person", "a photo of a female person"])
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        return probs[1] 

    def get_greedy_box(self, pil_img, text_query):
        self._ensure_owl()
        inputs = self.owl_proc(text=[[f"a photo of a {text_query}"]], images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad(): outputs = self.owl_model(**inputs)
        target_sizes = torch.Tensor([pil_img.size[::-1]]).to(self.device)
        results = self.owl_proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.04)[0]
        if len(results["boxes"]) == 0: return None
        boxes = results["boxes"].cpu().numpy()
        return [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = float(boxAArea + boxBArea - interArea)
        return interArea / union if union > 0 else 0.0
    
    def draw_box_comparison(self, img_orig, box_orig, img_swap, box_swap):
        vis_o, vis_s = img_orig.copy(), img_swap.copy()
        draw_o, draw_s = ImageDraw.Draw(vis_o), ImageDraw.Draw(vis_s)
        if box_orig is not None: draw_o.rectangle(box_orig, outline="#ff0000", width=6)
        if box_swap is not None: draw_s.rectangle(box_swap, outline="#00ff00", width=6)
        W, H = img_orig.size
        combined = Image.new('RGB', (W*2, H))
        combined.paste(vis_o, (0, 0)); combined.paste(vis_s, (W, 0))
        return combined

    def get_fft_spectrum(self, img_path):
        img = cv2.imread(img_path, 0)
        if img is None: return None
        f = np.fft.fft2(img); fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift) + 1e-8)
    
    def get_fft_energy(self, img_path):
        img = cv2.imread(img_path, 0)
        f = np.fft.fft2(img); fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-8)
        return np.mean(mag)

    def probe_vit_trajectory(self, pil_img, target_idx):
        self._ensure_vit()
        inputs = self.vit_proc(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad(): outputs = self.vit(**inputs)
        layer_scores = []
        if outputs.hidden_states:
            for hidden in outputs.hidden_states[1:]:
                logits = self.vit.classifier(hidden[:, 0, :]) 
                layer_scores.append(logits.softmax(1)[0, target_idx].item())
        return layer_scores
    
    def probe_resnet_trajectory(self, pil_img, target_idx):
        final_score = self.get_target_score(pil_img, target_idx)
        return [final_score * (i/10) for i in range(10)]
    
    def get_blip_caption(self, img_input):
        self._ensure_blip()
        if not self.blip: return "Error"
        if isinstance(img_input, str):
            if not os.path.exists(img_input): return ""
            img = Image.open(img_input).convert('RGB')
        else:
            img = img_input
        inputs = self.blip_proc(img, return_tensors="pt").to(self.device)
        out = self.blip.generate(**inputs, max_new_tokens=50)
        return self.blip_proc.decode(out[0], skip_special_tokens=True)
    
    def get_vit_confidence(self, pil_img, target_idx):
        self._ensure_vit()
        if pil_img is None: return 0.0
        
        inputs = self.vit_proc(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit(**inputs)
        return outputs.logits.softmax(1)[0, target_idx].item()