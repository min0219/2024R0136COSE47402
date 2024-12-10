import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import PIL
import torchvision.transforms as T
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DETRModule(nn.Module):
    """
    DETR을 사용한 객체 탐지 모듈
    """
    def __init__(self):
        super().__init__()
        # DETR 모델 로드
        self.detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.detr.eval()
        self.detr.to('cuda')
        
        # CLIP 모델 로드 (객체-텍스트 유사도 계산용)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 이미지 전처리
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def visualize_detection(self, image, boxes, scores, save_path="detr_detection.png"):
        """모든 탐지된 객체와 스코어를 시각화"""
        plt.figure(figsize=(12, 12))
        
        # PIL Image를 numpy 배열로 변환
        if isinstance(image, PIL.Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        plt.imshow(image_np)
        
        # 이미지 크기
        H, W = image_np.shape[:2]
        
        # 모든 박스 시각화
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.cpu() * torch.tensor([W, H, W, H])
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
            
            # 박스 그리기
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # 스코어 표시
            plt.text(x1, y1, f'{score:.2f}', 
                    bbox=dict(facecolor='white', alpha=0.7),
                    fontsize=12, color='red')
        
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    @torch.no_grad()
    def forward(self, image, target_text):
        # 이미지 전처리
        img = self.transform(image).unsqueeze(0).to('cuda')
        
        # DETR로 객체 탐지
        outputs = self.detr(img)
        
        # 예측된 박스와 로짓
        pred_boxes = outputs['pred_boxes'][0].to('cuda')
        pred_logits = outputs['pred_logits'][0].to('cuda')
        
        # 확률이 높은 박스만 선택
        probas = pred_logits.softmax(-1)[..., :-1]
        keep = probas.max(-1).values > 0.7
        
        # 관련 박스 선택
        relevant_boxes = pred_boxes[keep]

        scores = probas.max(-1).values
        relevant_scores = scores[keep]
        self.visualize_detection(image, relevant_boxes, relevant_scores)
        
        # CLIP으로 텍스트와의 관련성 계산
        if len(relevant_boxes) > 0:
            # 박스 영역을 이미지에서 추출
            crops = self.extract_crops(image, relevant_boxes)
            
            # CLIP으로 텍스트와 이미지 패치의 유사도 계산
            similarities = self.compute_clip_similarity(crops, target_text)
            
            # 가장 유사한 박스 선택
            best_box_idx = similarities.argmax()
            best_box = relevant_boxes[best_box_idx]
            
            return best_box.unsqueeze(0)
        
        return None

    def extract_crops(self, image, boxes):
        """박스 영역을 이미지에서 추출"""
        crops = []
        W, H = image.size
        for box in boxes:
            # 박스 좌표를 이미지 크기에 맞게 변환
            x1, y1, x2, y2 = box.cpu() * torch.tensor([W, H, W, H])
            
            # 좌표가 올바른 순서가 되도록 정렬
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 좌표가 이미지 범위를 벗어나지 않도록 클리핑
            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))
            
            # 최소 크기 보장
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
                
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            crops.append(crop)
        return crops

    def compute_clip_similarity(self, crops, text):
        """CLIP을 사용하여 이미지 패치와 텍스트의 유사도 계산"""
        # 텍스트 임베딩
        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        text_features = self.clip.get_text_features(**text_inputs.to('cuda'))
        
        similarities = []
        for crop in crops:
            # 이미지 패치 임베딩
            image_inputs = self.clip_processor(images=crop, return_tensors="pt", padding=True)
            image_features = self.clip.get_image_features(**image_inputs.to('cuda'))
            
            # 유사도 계산
            similarity = F.cosine_similarity(text_features, image_features)
            similarities.append(similarity.item())
        
        return torch.tensor(similarities).to('cuda')

class DETRModule2(nn.Module):
    def __init__(self):
        super().__init__()
        # DETR 모델 로드
        self.detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.detr.eval()
        self.detr.to('cuda')
        
        # CLIP 모델 로드
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 이미지 전처리
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # COCO 클래스
        self.CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox.device)
        return b

    def visualize_detection(self, image, prob, boxes, save_path="detr_detection.png"):
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        ax = plt.gca()
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
        
        for p, (xmin, ymin, xmax, ymax), color in zip(prob, boxes.tolist(), colors):
            # 박스 그리기
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     fill=False, color=color, linewidth=3))
            # 클래스 이름과 신뢰도 점수 표시
            cl = p.argmax()
            text = f'{self.CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin-10, text, 
                   bbox=dict(facecolor='white', alpha=0.7),
                   fontsize=10, color=color)

        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    @torch.no_grad()
    def forward(self, image, target_text):
        # 이미지 전처리
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
                
        # 원본 이미지 크기 저장
        W, H = image.size
        
        # DETR 입력을 위한 이미지 변환
        img = self.transform(image).unsqueeze(0).to('cuda')
        
        # DETR로 객체 탐지
        outputs = self.detr(img)
        
        # 예측 확률 계산 (no object 클래스 제외)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9
        
        # 박스 좌표 변환
        boxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
        
        if len(boxes_scaled) > 0:
            # 탐지 결과 시각화
            self.visualize_detection(image, probas[keep], boxes_scaled)
            
            # 텍스트와 가장 관련있는 객체 찾기
            crops = self.extract_crops(image, boxes_scaled)
            if crops:
                similarities = self.compute_clip_similarity(crops, target_text)
                best_box_idx = similarities.argmax()
                best_box = boxes_scaled[best_box_idx]
                
                # 박스 좌표를 0-1 범위로 정규화
                normalized_box = best_box / torch.tensor([W, H, W, H]).to(best_box.device)
                return normalized_box.unsqueeze(0)
        
        return None

    def extract_crops(self, image, boxes):
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.cpu().tolist())
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        return crops

    def compute_clip_similarity(self, crops, text):
        similarities = []
        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        text_features = self.clip.get_text_features(**text_inputs.to('cuda'))
        
        for crop in crops:
            image_inputs = self.clip_processor(images=crop, return_tensors="pt", padding=True)
            image_features = self.clip.get_image_features(**image_inputs.to('cuda'))
            similarity = F.cosine_similarity(text_features, image_features)
            similarities.append(similarity.item())
        
        return torch.tensor(similarities).to('cuda')
    
class MaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.detr = DETRModule2()
        
    def forward(self, image, text):
        # DETR로 관련 객체의 박스 얻기
        box = self.detr(image, text)
        
        if box is None:
            # 객체를 찾지 못한 경우 전체 이미지 마스크 반환
            return torch.zeros((1, 512, 512)).to('cuda')
        
        # 박스를 마스크로 변환
        mask = self.box_to_mask(box, image.size)
        return mask

    def box_to_mask(self, box, image_size):
        """박스 좌표를 마스크로 변환"""
        W, H = image_size
        mask = torch.zeros((1, 512, 512), device='cuda')  # 배치 차원 추가
        
        # 박스 좌표 변환 및 정규화
        x1, y1, x2, y2 = box[0].cpu() * torch.tensor([W, H, W, H])
        
        # 512x512 크기에 맞게 스케일링
        x1 = int((x1 * 512) / W)
        x2 = int((x2 * 512) / W)
        y1 = int((y1 * 512) / H)
        y2 = int((y2 * 512) / H)
        
        # 좌표가 올바른 순서가 되도록 정렬
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 좌표가 이미지 범위를 벗어나지 않도록 클리핑
        x1 = max(0, min(511, x1))
        x2 = max(0, min(511, x2))
        y1 = max(0, min(511, y1))
        y2 = max(0, min(511, y2))
        
        # 마스크 생성 (객체 영역을 1로 설정)
        mask[0, y1:y2+1, x1:x2+1] = 1
        
        return mask
    
class ImageEditor(nn.Module):
    """
    통합 이미지 편집 파이프라인
    """
    def __init__(self):
        super().__init__()
        self.mask_generator = MaskGenerator()
        self.stable_diffusion = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16
        ).to("cuda")

    def preprocess_image(self, image):
        """이미지 전처리"""
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
        elif isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image)
        return image

    def postprocess_mask(self, mask):
        """마스크 후처리"""
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)  # 배치 차원 제거
        
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask, mode='L')
    
    def visualize_mask(self, mask, save_path="mask_debug.png"):
        """마스크 시각화 및 저장"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

    #@torch.no_grad()
    @torch.inference_mode()
    def edit_image(self, image, edit_prompt):
        """이미지 편집 실행"""
        # 이미지 전처리
        image = self.preprocess_image(image)
        image = image.resize((512, 512))
        image.save("original_image.png")

        mask_prompt = edit_prompt[:edit_prompt.find("changes")]
        target_prompt = edit_prompt[edit_prompt.find("changes")+len("changes to"):]
        print(mask_prompt)
        print(target_prompt)
        # 마스크 생성
        mask = self.mask_generator(image, mask_prompt)
        self.visualize_mask(mask, "debug_mask.png")
        
        mask = self.postprocess_mask(mask)
        
        # Stable Diffusion으로 편집
        edited_image = self.stable_diffusion(
            prompt=target_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=40,
            guidance_scale=7.5,
            strength=0.9
        ).images[0]
        
        return edited_image
