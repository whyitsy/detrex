import torch
import cv2
import numpy as np
import json
from torchvision import transforms
from GoogleNet.model import build_model
from paddleocr import PaddleOCR


def load_class_indices(class_indices_path):
    """加载类别索引映射"""
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    return {int(k): v for k, v in class_indices.items()}

def load_image_with_opencv(image_path):
    """使用OpenCV加载图像，支持中文路径"""
    # 处理中文路径
    with open(image_path, 'rb') as f:
        img_data = f.read()
    
    # 将字节流转换为numpy数组
    img_array = np.frombuffer(img_data, np.uint8)
    
    # 使用cv2.imdecode解码图像
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    return image

def preprocess_image(opencv_image):
    """使用OpenCV图像进行预处理"""
    # 转换BGR到RGB
    image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    # 定义图像预处理变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换并添加batch维度
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def extract_ocr_text_with_ocr_instance(opencv_image, ocr_instance=None):
    """
    使用指定的OCR实例从OpenCV图像中提取文本
    """
    try:
        if ocr_instance is None:
            ocr_instance = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True
            )
        
        result = ocr_instance.predict(input=opencv_image)
        combined_text = ''
        
        for res in result:
            json_result = res.json
            rec_texts = json_result['res']['rec_texts']
            combined_text = ''.join(rec_texts)

        return combined_text
        
    except Exception as e:
        print(f"OCR提取失败: {e}")
        return ""

def predict_image(model, image_tensor, ocr_text, class_indices, device):
    """预测图像类别"""
    model.eval()
    
    with torch.no_grad():
        # 将数据移到指定设备
        image_tensor = image_tensor.to(device)
        
        # 进行预测
        outputs = model(image_tensor, [ocr_text])
        
        # 计算概率
        probabilities = torch.softmax(outputs, dim=1)
        
        # 获取预测类别和置信度
        confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # 获取类别名称
        class_name = class_indices.get(predicted_class, f"未知类别_{predicted_class}")
        
        return predicted_class, class_name, confidence, probabilities[0]

class ImagePredictor:
    """
    图像预测器类，支持模型和OCR的复用
    """
    
    def __init__(self, config_path='GoogleNet/config.json', model_path=None):
        """初始化预测器"""
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载类别索引
        self.class_indices = load_class_indices(self.config['class_indices_path'])
        self.num_classes = len(self.class_indices)
        
        # 设置设备
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 构建和加载模型
        self.model = build_model(config_path, self.num_classes)
        if model_path is None:
            model_path = self.config['model_save_path']
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        
        # 初始化OCR
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True
        )
    
    def predict(self, opencv_image):
        """预测单个图像"""
        # 预处理图像
        image_tensor = preprocess_image(opencv_image)
        
        # 提取OCR文本 - 使用修复后的函数
        ocr_text = extract_ocr_text_with_ocr_instance(opencv_image, self.ocr)
        
        # 进行预测
        predicted_class, class_name, confidence, all_probabilities = predict_image(
            self.model, image_tensor, ocr_text, self.class_indices, self.device
        )
        
        return {
            'predicted_class_id': predicted_class,
            'predicted_class_name': class_name,
            'confidence': confidence,
            'ocr_text': ocr_text
        }
    
    def predict_batch(self, opencv_images):
        """批量预测多个图像"""
        results = []
        for i, opencv_image in enumerate(opencv_images):
            try:
                result = self.predict(opencv_image)
                result['image_index'] = i
                result['success'] = True
                results.append(result)
            except Exception as e:
                results.append({
                    'image_index': i,
                    'error': str(e),
                    'success': False
                })
        return results

