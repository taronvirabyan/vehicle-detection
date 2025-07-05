"""
Конвертер видео для обеспечения совместимости с веб-браузерами
"""

import cv2
import numpy as np
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


def convert_video_for_web(input_path, output_path=None, max_width=1280, target_fps=30):
    """
    Конвертирует видео в веб-совместимый формат
    
    Args:
        input_path: Путь к входному видео
        output_path: Путь к выходному видео (если None, создается временный файл)
        max_width: Максимальная ширина видео
        target_fps: Целевой FPS
        
    Returns:
        Путь к конвертированному видео
    """
    
    # Открываем входное видео
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео: {input_path}")
        return None
    
    # Получаем параметры
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Рассчитываем новые размеры
    if orig_width > max_width:
        scale = max_width / orig_width
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    else:
        new_width = orig_width
        new_height = orig_height
    
    # Убеждаемся что размеры четные
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    # Выходной путь
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    # Создаем writer с XVID кодеком (универсальный)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))
    
    if not out.isOpened():
        logger.error("Не удалось создать VideoWriter")
        cap.release()
        return None
    
    logger.info(f"Конвертация видео: {orig_width}x{orig_height} -> {new_width}x{new_height}, "
               f"{orig_fps} -> {target_fps} FPS")
    
    # Обработка кадров
    frame_count = 0
    skip_ratio = orig_fps / target_fps if orig_fps > target_fps else 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Пропускаем кадры для уменьшения FPS
        if skip_ratio > 1 and frame_count % int(skip_ratio) != 0:
            frame_count += 1
            continue
        
        # Изменяем размер если нужно
        if new_width != orig_width:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Записываем кадр
        out.write(frame)
        frame_count += 1
        
        # Прогресс
        if frame_count % 100 == 0:
            logger.info(f"Обработано кадров: {frame_count}")
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    
    logger.info(f"✅ Видео конвертировано: {output_path}")
    
    # Проверяем размер файла
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Размер файла: {size_mb:.1f} MB")
        
        if size_mb > 100:
            logger.warning("⚠️ Файл слишком большой для веб. Рекомендуется обрезать видео.")
    
    return output_path


def create_preview_video(input_path, duration_sec=30, start_sec=0):
    """
    Создает короткое превью видео для быстрой демонстрации
    
    Args:
        input_path: Путь к входному видео
        duration_sec: Длительность превью
        start_sec: Начальная позиция
        
    Returns:
        Путь к превью видео
    """
    
    temp_file = tempfile.NamedTemporaryFile(suffix='_preview.mp4', delete=False)
    output_path = temp_file.name
    temp_file.close()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Уменьшаем разрешение для превью
    if width > 854:  # 480p
        scale = 854 / width
        width = int(width * scale)
        height = int(height * scale)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
    
    # Переход к начальной позиции
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    
    # Создаем writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Обрабатываем кадры
    frames_to_process = int(duration_sec * fps)
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Изменяем размер
        frame = cv2.resize(frame, (width, height))
        
        # Добавляем информацию о превью
        cv2.putText(frame, f"PREVIEW: {i//fps}s / {duration_sec}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    logger.info(f"✅ Превью создано: {output_path}")
    return output_path