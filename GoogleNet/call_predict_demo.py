import cv2
from predict import ImagePredictor
    
# 初始化预测器
predictor = ImagePredictor()
    
# 读取图像
image = cv2.imread('/mnt/data/datasets/shampoo_FGVCdataset_low_quality_2/test/100年润发净爽去屑洗发露480ml/22.jpg')
    
# 进行预测
result = predictor.predict(image)
print(f"预测结果: {result}")
'''
预测结果: {'predicted_class_id': 0, 'predicted_class_name': '100年润发净爽去屑洗发露480ml', 'confidence': 0.9983069896697998, 'ocr_text': ''}
'''    
