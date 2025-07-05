"""
Утилиты для работы с видео
"""

import cv2
import logging
import platform

logger = logging.getLogger(__name__)


def get_video_writer_fourcc():
    """
    Определяет лучший доступный fourcc для браузер-совместимого видео
    
    Returns:
        fourcc код для cv2.VideoWriter
    """
    system = platform.system()
    
    # Приоритетный список кодеков для веб-совместимости
    # ВАЖНО: НЕ используйте XVID - он не поддерживается браузерами!
    codecs_to_try = [
        'avc1',  # H.264 - лучший для веб, поддерживается всеми браузерами
        'h264',  # Альтернативное имя H.264
        'x264',  # x264 encoder
        'mp4v',  # MPEG-4 Part 2 (резервный вариант, ограниченная поддержка)
    ]
    
    # Для Windows может потребоваться другой порядок
    if system == 'Windows':
        codecs_to_try = ['h264', 'avc1', 'x264', 'mp4v']
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            logger.info(f"Используется видео кодек: {codec}")
            return fourcc
        except Exception as e:
            logger.warning(f"Кодек {codec} недоступен: {e}")
    
    # Если ничего не работает, используем системный дефолт
    logger.warning("Используется дефолтный кодек (-1)")
    return -1


def create_video_writer(output_path, fps, frame_size):
    """
    Создает VideoWriter с оптимальными настройками для веб-воспроизведения
    
    Args:
        output_path: Путь к выходному файлу
        fps: Частота кадров
        frame_size: Размер кадра (width, height)
        
    Returns:
        cv2.VideoWriter объект или None при ошибке
    """
    try:
        # Получаем оптимальный fourcc
        fourcc = get_video_writer_fourcc()
        
        # Создаем writer
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not out.isOpened():
            logger.error(f"Не удалось создать VideoWriter для {output_path}")
            return None
            
        logger.info(f"VideoWriter создан: {output_path}, {frame_size}, {fps} FPS")
        return out
        
    except Exception as e:
        logger.error(f"Ошибка создания VideoWriter: {e}")
        return None


def ensure_even_dimensions(width, height):
    """
    Обеспечивает четные размеры видео (требование многих кодеков)
    
    Args:
        width: Ширина видео
        height: Высота видео
        
    Returns:
        (width, height) - четные размеры
    """
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    return width, height