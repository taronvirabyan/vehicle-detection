"""
Утилиты для создания браузер-совместимых видео
"""

import cv2
import numpy as np
import logging
import platform
import os

logger = logging.getLogger(__name__)


def create_browser_compatible_video_writer(output_path, fps, frame_size, use_h264=True):
    """
    Создает VideoWriter с гарантированной поддержкой в браузерах
    
    Args:
        output_path: Путь к выходному файлу (.mp4 или .webm)
        fps: Частота кадров
        frame_size: Размер кадра (width, height)
        use_h264: Использовать H.264 (True) или VP8/WebM (False)
        
    Returns:
        cv2.VideoWriter объект или None при ошибке
    """
    width, height = frame_size
    
    # Убедимся, что размеры четные (требование для многих кодеков)
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    frame_size = (width, height)
    
    if use_h264 or output_path.endswith('.mp4'):
        # H.264 для MP4 - самый универсальный вариант
        logger.info("Используется H.264 кодек для браузерной совместимости")
        
        # Пробуем разные варианты H.264
        h264_codecs = [
            'avc1',  # Основной H.264
            'h264',  # Альтернативное имя
            'x264',  # x264 encoder
            'H264',  # Еще один вариант
        ]
        
        for codec in h264_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                
                if out.isOpened():
                    logger.info(f"✅ Успешно создан VideoWriter с кодеком: {codec}")
                    return out
                else:
                    logger.warning(f"Не удалось открыть VideoWriter с кодеком: {codec}")
                    
            except Exception as e:
                logger.warning(f"Ошибка с кодеком {codec}: {e}")
        
        # Если не сработали специфичные кодеки, пробуем mp4v
        logger.warning("H.264 кодеки не доступны, пробуем MP4V")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if out.isOpened():
            logger.info("✅ Используется fallback кодек MP4V")
            return out
            
    else:
        # VP8 для WebM - альтернативный открытый формат
        logger.info("Используется VP8 кодек для WebM")
        
        webm_codecs = ['VP80', 'VP90']
        
        for codec in webm_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                
                if out.isOpened():
                    logger.info(f"✅ Успешно создан VideoWriter с кодеком: {codec}")
                    return out
                    
            except Exception as e:
                logger.warning(f"Ошибка с кодеком {codec}: {e}")
    
    logger.error("❌ Не удалось создать браузер-совместимый VideoWriter")
    return None


def convert_to_browser_compatible(input_path, output_path=None):
    """
    Конвертирует видео в браузер-совместимый формат
    
    Args:
        input_path: Путь к исходному видео
        output_path: Путь для сохранения (если None, заменяет расширение на .mp4)
        
    Returns:
        Путь к конвертированному файлу или None при ошибке
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_browser.mp4"
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео: {input_path}")
        return None
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Создаем браузер-совместимый writer
    out = create_browser_compatible_video_writer(output_path, fps, (width, height))
    
    if out is None:
        cap.release()
        return None
    
    logger.info(f"Конвертация: {input_path} -> {output_path}")
    logger.info(f"Параметры: {width}x{height}, {fps} FPS, {frame_count} кадров")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            progress = (frame_idx / frame_count) * 100
            logger.info(f"Прогресс: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    logger.info(f"✅ Конвертация завершена: {output_path}")
    return output_path


# Пример использования для замены XVID
def example_usage():
    """
    Пример создания видео, которое гарантированно работает в браузере
    """
    width, height = 640, 480
    fps = 30
    duration = 5  # секунд
    
    # Создаем видео с H.264 кодеком
    output_path = "browser_compatible_video.mp4"
    out = create_browser_compatible_video_writer(output_path, fps, (width, height))
    
    if out is None:
        logger.error("Не удалось создать VideoWriter")
        return
    
    # Генерируем простое тестовое видео
    for i in range(fps * duration):
        # Создаем кадр с градиентом
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Анимированный градиент
        color_value = int((i / (fps * duration)) * 255)
        frame[:, :] = [color_value, 255 - color_value, 128]
        
        # Добавляем текст
        text = f"Frame {i+1}/{fps*duration}"
        cv2.putText(frame, text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Движущийся круг
        circle_x = int((i / (fps * duration)) * width)
        cv2.circle(frame, (circle_x, height//2), 30, (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"✅ Видео создано: {output_path}")
    

# Функция для проверки поддержки кодеков
def check_codec_support():
    """
    Проверяет доступные кодеки для создания браузер-совместимых видео
    """
    test_codecs = {
        'H.264': ['avc1', 'h264', 'x264', 'H264'],
        'MPEG-4': ['mp4v', 'MP4V'],
        'WebM': ['VP80', 'VP90'],
        'XVID': ['XVID', 'xvid'],  # Не работает в браузере!
    }
    
    width, height = 640, 480
    results = {}
    
    for format_name, codecs in test_codecs.items():
        results[format_name] = {}
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_file = f"test_{codec}.mp4" if format_name != 'WebM' else f"test_{codec}.webm"
                out = cv2.VideoWriter(temp_file, fourcc, 30, (width, height))
                
                if out.isOpened():
                    # Записываем один кадр для проверки
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(frame)
                    out.release()
                    
                    # Проверяем, что файл создан
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        results[format_name][codec] = "✅ Работает"
                        os.remove(temp_file)
                    else:
                        results[format_name][codec] = "❌ Файл не создан"
                else:
                    results[format_name][codec] = "❌ Writer не открылся"
                    
            except Exception as e:
                results[format_name][codec] = f"❌ Ошибка: {str(e)}"
    
    # Выводим результаты
    print("\n=== Проверка поддержки кодеков ===")
    print("Браузеры поддерживают: H.264, WebM (VP8/VP9)")
    print("НЕ поддерживают: XVID, DivX, и другие старые кодеки\n")
    
    for format_name, codecs in results.items():
        print(f"\n{format_name}:")
        for codec, status in codecs.items():
            browser_support = "🌐 Браузер ✅" if format_name in ['H.264', 'WebM'] else "🌐 Браузер ❌"
            print(f"  {codec}: {status} | {browser_support}")
    
    return results


if __name__ == "__main__":
    # Проверяем поддержку кодеков
    check_codec_support()
    
    # Создаем пример браузер-совместимого видео
    print("\n\n=== Создание тестового видео ===")
    example_usage()