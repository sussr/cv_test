"""
Точка входа для детекции людей в видео.

Программа читает входной видеофайл, выполняет детекцию людей с помощью
MediaPipe Object Detector и сохраняет результат в новый видеофайл
с прямоугольниками и подписью (класс + уверенность).
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


def visualize_frame(frame: np.ndarray, detection_result) -> np.ndarray:
    """
    Наносит на кадр прямоугольники вокруг обнаруженных объектов
    и подписи с классом и уверенностью.

    Args:
        frame: Исходный кадр BGR (NumPy array).
        detection_result: Результат работы ObjectDetector (детекции).

    Returns:
        Кадр с отрисованными прямоугольниками и подписями.
    """
    BOX_COLOR = (0, 255, 0)    # Зеленый
    TEXT_COLOR = (0, 0, 255)   # Красный
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS = 1
    MARGIN = 5

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2 = x1 + int(bbox.width)
        y2 = y1 + int(bbox.height)

        # Рисуем рамку
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness=2)

        # Готовим подпись: "person (0.85)"
        category = detection.categories[0]
        label = f"{category.category_name} ({category.score:.2f})"
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)[0]

        # Корректируем позицию подписи, чтобы она не ушла за рамки кадра
        text_x = x1
        text_y = y1 - MARGIN
        if text_y - text_size[1] < 0:
            text_y = y1 + text_size[1] + MARGIN

        # Рисуем подпись
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def run_detection(
    input_video_path: str,
    output_video_path: str,
    model_path: str,
    score_threshold: float = 0.2,
):
    """
    Основная функция: открывает видео, выполняет детекцию по кадрам и сохраняет результат.

    Args:
        input_video_path: Путь к исходному видеофайлу.
        output_video_path: Путь, куда сохранить аннотированное видео.
        model_path: Путь к TFLite-модели (например, "efficientdet_lite2.tflite").
        score_threshold: Порог уверенности детектора (от 0 до 1).
    """
    # Проверка существования входного файла
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Не найден входной файл: {input_video_path}")

    # Открываем видео для чтения
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Создаем VideoWriter для записи результата
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать VideoWriter: {output_video_path}")

    # Настройка детектора MediaPipe
    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        score_threshold=score_threshold,
        max_results=60,
        category_allowlist=["person"],
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Обрабатываем каждый кадр
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертируем кадр BGR->RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Временная метка кадра в микросекундах
        timestamp_us = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_us)

        # Отрисовываем детекции и записываем кадр
        annotated_frame = visualize_frame(frame, result)
        writer.write(annotated_frame)

    # Освобождаем ресурсы
    cap.release()
    writer.release()
    detector.close()


def main():
    """
    Точка входа в программу. Парсит константы и запускает функцию детекции.
    """
    # Пути к файлам
    INPUT_VIDEO = r"crowd.mp4"
    OUTPUT_VIDEO = "annotated.mp4"
    MODEL_FILE = r"efficientdet_lite2.tflite"

    # Запуск детекции
    run_detection(
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        model_path=MODEL_FILE,
        score_threshold=0.2,
    )
    print(f"Видео с отрисованными людьми сохранено в '{OUTPUT_VIDEO}'.")


if __name__ == "__main__":
    main()
