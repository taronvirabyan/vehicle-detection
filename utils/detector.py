from ultralytics import YOLO
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    Надежный детектор транспортных средств на основе YOLOv8.
    Оптимизирован для максимальной стабильности и производительности.
    """
    
    def __init__(
        self, 
        model_path: str = 'yolov8n.pt',
        confidence: float = 0.45,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Инициализация детектора.
        
        Args:
            model_path: Путь к весам модели
            confidence: Минимальная уверенность для детекции
            iou_threshold: Порог IoU для NMS
            device: Устройство для инференса ('cpu' или 'cuda')
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Классы транспортных средств в COCO
        self.vehicle_classes = {
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7
        }
        self.vehicle_class_ids = set(self.vehicle_classes.values())
        
        # Определение устройства
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Использование устройства: {self.device}")
        
        # Загрузка модели с обработкой ошибок
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Прогрев модели для стабильной производительности
            self._warmup_model()
            
            logger.info(f"Модель {model_path} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def _warmup_model(self):
        """Прогрев модели для стабильной производительности"""
        try:
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(3):
                self.model(dummy_img, verbose=False)
            logger.info("Прогрев модели завершен")
        except Exception as e:
            logger.warning(f"Ошибка при прогреве модели: {e}")
    
    def detect_vehicles(
        self, 
        frame: np.ndarray,
        return_annotated: bool = False
    ) -> Tuple[bool, List[Dict], Optional[np.ndarray]]:
        """
        Детекция транспортных средств на кадре.
        
        Args:
            frame: Кадр для обработки (BGR)
            return_annotated: Вернуть аннотированный кадр
            
        Returns:
            (has_vehicles, vehicles_list, annotated_frame)
        """
        try:
            # Валидация входных данных
            if frame is None or frame.size == 0:
                logger.warning("Пустой кадр")
                return False, [], None
            
            # Выполнение детекции
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                classes=list(self.vehicle_class_ids)
            )
            
            vehicles = []
            
            for r in results:
                if r.boxes is None:
                    continue
                    
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if class_id not in self.vehicle_class_ids:
                        continue
                    
                    # Получение имени класса
                    class_name = self.model.names[class_id]
                    
                    # Извлечение координат
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    vehicle_info = {
                        'class': class_name,
                        'class_id': class_id,
                        'confidence': float(box.conf[0]),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    
                    vehicles.append(vehicle_info)
            
            # Сортировка по уверенности
            vehicles.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Аннотированный кадр
            annotated_frame = None
            if return_annotated and len(results) > 0:
                annotated_frame = results[0].plot()
            
            has_vehicles = len(vehicles) > 0
            
            return has_vehicles, vehicles, annotated_frame
            
        except Exception as e:
            logger.error(f"Ошибка при детекции: {e}")
            return False, [], None
    
    def set_confidence(self, confidence: float):
        """Изменение порога уверенности"""
        if 0.0 <= confidence <= 1.0:
            self.confidence = confidence
            logger.info(f"Порог уверенности изменен на {confidence}")
        else:
            logger.warning(f"Недопустимое значение confidence: {confidence}")
    
    def get_stats(self) -> Dict:
        """Получение статистики модели"""
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'yolov8n',
            'device': self.device,
            'confidence_threshold': self.confidence,
            'iou_threshold': self.iou_threshold,
            'vehicle_classes': list(self.vehicle_classes.keys())
        }