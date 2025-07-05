import cv2
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional, Callable
import logging
from pathlib import Path
import time
import subprocess
import os
import tempfile
from .video_utils import create_video_writer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Процессор видео с оптимизацией производительности и надежной обработкой ошибок.
    """
    
    def __init__(
        self,
        detector,
        skip_frames: int = 2,
        history_size: int = 15,
        detection_threshold: float = 0.4,
        max_width: int = 640
    ):
        """
        Args:
            detector: Экземпляр VehicleDetector
            skip_frames: Количество пропускаемых кадров
            history_size: Размер буфера для сглаживания
            detection_threshold: Порог для определения наличия транспорта
            max_width: Максимальная ширина кадра
        """
        self.detector = detector
        self.skip_frames = skip_frames
        self.history_size = history_size
        self.detection_threshold = detection_threshold
        self.max_width = max_width
        
        # История детекций для сглаживания результатов
        self.detection_history = deque(maxlen=history_size)
        
        # Статистика
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'vehicles_detected': 0,
            'processing_time': 0
        }
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Изменение размера кадра с сохранением пропорций"""
        if frame is None:
            return frame
            
        height, width = frame.shape[:2]
        if width > self.max_width:
            scale = self.max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame
    
    def draw_results(
        self,
        frame: np.ndarray,
        vehicles: List[Dict],
        vehicle_present: bool,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Отрисовка результатов детекции на кадре"""
        if frame is None:
            return frame
            
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Цветовая схема
        color_present = (0, 255, 0)  # Зеленый
        color_absent = (0, 0, 255)   # Красный
        color_box = (255, 255, 0)    # Желтый для bbox
        
        # Рисуем bbox для каждого транспортного средства
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle['bbox'])
            
            # Ограничение координат
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            
            # Прямоугольник
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_box, 2)
            
            # Метка с классом и уверенностью
            label = f"{vehicle['class']} {vehicle['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Фон для текста
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color_box,
                -1
            )
            
            # Текст
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        # Статус детекции
        status = "VEHICLE DETECTED" if vehicle_present else "NO VEHICLE"
        status_color = color_present if vehicle_present else color_absent
        
        # Фон для статуса
        cv2.rectangle(annotated, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (300, 50), status_color, 2)
        
        # Текст статуса
        cv2.putText(
            annotated,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )
        
        # FPS если доступен
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                annotated,
                fps_text,
                (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Количество обнаруженных объектов
        count_text = f"Count: {len(vehicles)}"
        cv2.putText(
            annotated,
            count_text,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return annotated
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Tuple[bool, List[Dict], np.ndarray, bool]:
        """
        Обработка одного кадра.
        
        Returns:
            (should_process, vehicles, annotated_frame, vehicle_present)
        """
        # Пропуск кадров для оптимизации
        should_process = frame_idx % (self.skip_frames + 1) == 0
        
        if not should_process:
            # Используем последний результат
            vehicle_present = self._get_smoothed_detection()
            annotated = self.draw_results(frame, [], vehicle_present)
            return False, [], annotated, vehicle_present
        
        # Изменение размера для ускорения
        resized_frame = self.resize_frame(frame)
        
        # Детекция
        has_vehicle, vehicles, _ = self.detector.detect_vehicles(resized_frame)
        
        # Масштабирование bbox обратно
        if resized_frame.shape != frame.shape:
            scale_x = frame.shape[1] / resized_frame.shape[1]
            scale_y = frame.shape[0] / resized_frame.shape[0]
            
            for vehicle in vehicles:
                vehicle['bbox'] = [
                    vehicle['bbox'][0] * scale_x,
                    vehicle['bbox'][1] * scale_y,
                    vehicle['bbox'][2] * scale_x,
                    vehicle['bbox'][3] * scale_y
                ]
                vehicle['center'] = [
                    vehicle['center'][0] * scale_x,
                    vehicle['center'][1] * scale_y
                ]
        
        # Обновление истории
        self.detection_history.append(has_vehicle)
        
        # Сглаженный результат
        vehicle_present = self._get_smoothed_detection()
        
        # Отрисовка
        annotated_frame = self.draw_results(frame, vehicles, vehicle_present)
        
        # Обновление статистики
        if vehicle_present:
            self.stats['vehicles_detected'] += 1
        
        return True, vehicles, annotated_frame, vehicle_present
    
    def _get_smoothed_detection(self) -> bool:
        """Получение сглаженного результата детекции"""
        if not self.detection_history:
            return False
        
        detection_ratio = sum(self.detection_history) / len(self.detection_history)
        return detection_ratio >= self.detection_threshold
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Обработка видеофайла.
        
        Args:
            video_path: Путь к видео
            output_path: Путь для сохранения результата
            progress_callback: Callback для прогресса
            max_frames: Максимальное количество кадров
            
        Returns:
            Словарь с результатами и статистикой
        """
        start_time = time.time()
        
        # Проверка существования файла
        if not Path(video_path).exists():
            logger.error(f"Видеофайл не найден: {video_path}")
            return {'error': 'Video file not found'}
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return {'error': 'Failed to open video'}
        
        # Параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            frame_count = min(frame_count, max_frames)
        
        logger.info(f"Обработка видео: {width}x{height}, {fps} FPS, {frame_count} кадров")
        
        # Подготовка записи результата
        out_writer = None
        temp_video_path = None
        if output_path:
            # Создаем временное видео с mp4v
            temp_video_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        # Результаты
        processed_frames = []
        frame_idx = 0
        
        # Сброс статистики
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'vehicles_detected': 0,
            'processing_time': 0
        }
        self.detection_history.clear()
        
        try:
            while cap.isOpened() and frame_idx < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обработка кадра
                frame_start = time.time()
                processed, vehicles, annotated, vehicle_present = self.process_frame(frame, frame_idx)
                frame_time = time.time() - frame_start
                
                # Статистика
                self.stats['total_frames'] += 1
                if processed:
                    self.stats['processed_frames'] += 1
                    self.stats['processing_time'] += frame_time
                
                # Сохранение
                if out_writer:
                    out_writer.write(annotated)
                else:
                    processed_frames.append(annotated)
                
                # Прогресс
                if progress_callback:
                    progress = (frame_idx + 1) / frame_count
                    progress_callback(progress)
                
                frame_idx += 1
        
        except Exception as e:
            logger.error(f"Ошибка при обработке видео: {e}")
            return {'error': str(e)}
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
                
                # Конвертируем в H.264 для браузера если есть FFmpeg
                if temp_video_path and os.path.exists(temp_video_path):
                    if self._convert_to_h264(temp_video_path, output_path):
                        os.remove(temp_video_path)
                    else:
                        # Если конвертация не удалась, используем оригинал
                        os.rename(temp_video_path, output_path)
        
        # Финальная статистика
        total_time = time.time() - start_time
        avg_fps = self.stats['processed_frames'] / total_time if total_time > 0 else 0
        
        results = {
            'success': True,
            'frames': processed_frames if not output_path else [],
            'stats': {
                **self.stats,
                'total_time': total_time,
                'average_fps': avg_fps,
                'detection_rate': self.stats['vehicles_detected'] / self.stats['total_frames'] 
                    if self.stats['total_frames'] > 0 else 0
            }
        }
        
        logger.info(f"Обработка завершена: {avg_fps:.1f} FPS, "
                   f"{results['stats']['detection_rate']:.1%} кадров с транспортом")
        
        return results
    
    def reset(self):
        """Сброс состояния процессора"""
        self.detection_history.clear()
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'vehicles_detected': 0,
            'processing_time': 0
        }
    
    def _convert_to_h264(self, input_path: str, output_path: str) -> bool:
        """Конвертация видео в H.264 для совместимости с браузером"""
        try:
            # Проверяем наличие FFmpeg
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode != 0:
                logger.warning("FFmpeg не найден, используем mp4v кодек")
                return False
            
            # Конвертируем в H.264
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y', output_path
            ]
            
            logger.info("Конвертация видео в H.264 для браузера...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info("✅ Видео успешно конвертировано в H.264")
                return True
            else:
                logger.error(f"Ошибка конвертации: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при конвертации видео: {e}")
            return False