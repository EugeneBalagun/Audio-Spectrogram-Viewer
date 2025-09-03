import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QScrollArea, QLabel, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
import soundfile as sf
import os
import logging
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_audio_file(file_path):
    """
    Читает весь аудиофайл и возвращает сигнал и частоту дискретизации.
    """
    logging.info(f"Загрузка аудиофайла: {file_path}")
    y, sr = sf.read(file_path)
    if len(y.shape) > 1:  # Если стерео, берём только один канал
        y = y[:, 0]
    return y, sr

def standard_fft_spectrogram(signal, sample_rate, window_size, step_size):
    spectrogram = []
    window = np.hanning(window_size)

    for start in range(0, len(signal) - window_size, step_size):
        segment = signal[start:start + window_size] * window
        fft_result = np.fft.fft(segment)
        magnitude = np.abs(fft_result[:window_size // 2])
        spectrogram.append(magnitude)

    spectrogram = np.array(spectrogram).T
    time = np.arange(spectrogram.shape[1]) * (step_size / sample_rate)
    freq = np.fft.fftfreq(window_size, d=1/sample_rate)[:window_size // 2]

    return spectrogram, time, freq

def process_full_audio(signal, sample_rate, window_size, step_size, chunk_duration_sec):
    chunk_size = int(chunk_duration_sec * sample_rate)
    full_spectrogram = []
    full_time = []

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        if len(chunk) < window_size:
            break

        spectrogram, time, freq = standard_fft_spectrogram(chunk, sample_rate, window_size, step_size)
        if len(full_spectrogram) == 0:
            full_spectrogram = spectrogram
        else:
            full_spectrogram = np.hstack((full_spectrogram, spectrogram))
        if len(full_time) == 0:
            full_time = time + i / sample_rate
        else:
            full_time = np.concatenate((full_time, time + i / sample_rate))

    return full_spectrogram, full_time, freq

def create_spectrogram_image(spectrogram, time, freq, width_pixels, height_pixels=600):
    """
    Создаёт начальное изображение спектрограммы без линии воспроизведения.
    Возвращает изображение и границы области спектрограммы.
    """
    logging.info(f"Создание спектрограммы: размер={spectrogram.shape}, время={time.shape}, частоты={freq.shape}")
    if spectrogram.size == 0 or time.size == 0 or freq.size == 0:
        logging.error("Некорректные данные спектрограммы")
        return None, None

    # Динамические размеры фигуры
    fig_width = max(12, min(width_pixels / 100, 50))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    im = ax.imshow(
        20 * np.log10(spectrogram + 1e-6),
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='magma'
    )
    ax.set_xlabel('Час [с]')
    ax.set_ylabel('Частота [Гц]')
    ax.set_title("FFT спектрограмма")
    fig.colorbar(im, ax=ax, label='Амплітуда [dB]')

    # Уменьшаем отступы
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

    # Получаем границы области спектрограммы
    ax_pos = ax.get_position()
    data_area = {
        'x0': ax_pos.x0,
        'x1': ax_pos.x1,
        'width': ax_pos.width
    }
    logging.info(f"Границы области спектрограммы: x0={data_area['x0']:.2f}, x1={data_area['x1']:.2f}, width={data_area['width']:.2f}")

    # Рендеринг в изображение
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()
    image = QImage(buf, width, height, QImage.Format_RGBA8888)
    
    plt.close(fig)
    return image, data_area

def format_time(seconds):
    """Форматирует время в MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

class SpectrogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFT спектрограмма")
        self.setGeometry(100, 100, 800, 600)

        # Инициализация pygame для аудио
        pygame.mixer.init()

        # Основной виджет и компоновка
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Кнопки управления файлом
        self.file_button_layout = QHBoxLayout()
        self.load_button = QPushButton("Загрузить аудиофайл", self)
        self.load_button.clicked.connect(self.load_audio)
        self.file_button_layout.addWidget(self.load_button)

        self.save_button = QPushButton("Сохранить спектрограмму", self)
        self.save_button.clicked.connect(self.save_spectrogram)
        self.file_button_layout.addWidget(self.save_button)
        self.layout.addLayout(self.file_button_layout)

        # Кнопки управления воспроизведением
        self.playback_button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play/Pause", self)
        self.play_button.clicked.connect(self.toggle_playback)
        self.playback_button_layout.addWidget(self.play_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_playback)
        self.playback_button_layout.addWidget(self.stop_button)
        self.layout.addLayout(self.playback_button_layout)

        # Кнопки зума
        self.zoom_button_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In", self)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out", self)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_button_layout.addWidget(self.zoom_out_button)
        self.layout.addLayout(self.zoom_button_layout)

        # Индикатор времени
        self.time_label = QLabel("00:00 / 00:00", self)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.time_label)

        # Слайдер для перемотки
        self.seek_slider = QSlider(Qt.Horizontal, self)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(1000)  # Разрешение слайдера (0–1000)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self.seek_position)
        self.layout.addWidget(self.seek_slider)

        # Область прокрутки
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Метка для отображения спектрограммы
        self.spectrogram_label = QLabel(self)
        self.spectrogram_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.spectrogram_label)

        # Переменные
        self.audio_file = None
        self.spectrogram_image = None
        self.spectrogram = None
        self.time = None
        self.freq = None
        self.is_playing = False
        self.play_position = 0
        self.total_duration = 0
        self.image_width = 800
        self.position_buffer = deque(maxlen=5)
        self.log_counter = 0
        self.data_area = None
        self.zoom_factor = 1.0
        self.play_start_time = 0  # Время начала воспроизведения или последней перемотки

        # Таймер для обновления позиции воспроизведения
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_play_position)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудиофайл", "", "Audio files (*.mp3 *.wav *.flac)"
        )
        if not file_path:
            return
        self.audio_file = file_path

        # Читаем аудиофайл
        audio_data, sample_rate = read_audio_file(file_path)

        # Генерируем спектрограмму
        window_size = 1024
        step_size = 512
        chunk_duration_sec = 4
        self.spectrogram, self.time, self.freq = process_full_audio(
            audio_data, sample_rate, window_size, step_size, chunk_duration_sec
        )

        # Проверяем данные
        if self.spectrogram.size == 0 or self.time.size == 0:
            logging.error("Ошибка: пустая спектрограмма или временная ось")
            QMessageBox.critical(self, "Ошибка", "Не удалось сгенерировать спектрограмму!")
            return

        # Устанавливаем ширину изображения и длительность
        self.total_duration = self.time[-1]
        self.image_width = max(800, int(500 * self.total_duration / 10))
        self.zoom_factor = 1.0
        logging.info(f"Длительность аудио: {self.total_duration:.2f} секунд, ширина изображения: {self.image_width} пикселей")

        # Создаём начальное изображение спектрограммы
        self.spectrogram_image, self.data_area = create_spectrogram_image(
            self.spectrogram, self.time, self.freq, self.image_width
        )
        if self.spectrogram_image is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось создать изображение спектрограммы!")
            return

        self.spectrogram_label.setPixmap(QPixmap.fromImage(self.spectrogram_image))
        self.spectrogram_label.adjustSize()

        # Сбрасываем прокрутку
        self.scroll_area.horizontalScrollBar().setValue(0)
        logging.info("Прокрутка сброшена на начало")

        # Обновляем индикатор времени
        self.time_label.setText(f"00:00 / {format_time(self.total_duration)}")

        # Настраиваем слайдер
        self.seek_slider.setValue(0)
        logging.info("Слайдер сброшен на начало")

        # Загружаем аудио в pygame
        try:
            pygame.mixer.music.load(file_path)
            logging.info("Аудио успешно загружено в pygame")
        except pygame.error as e:
            logging.error(f"Ошибка загрузки аудио в pygame: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить аудио: {e}")
            self.audio_file = None
            return

    def toggle_playback(self):
        if not self.audio_file:
            QMessageBox.critical(self, "Ошибка", "Сначала загрузите аудиофайл!")
            return

        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.timer.stop()
            logging.info("Воспроизведение приостановлено")
        else:
            try:
                if pygame.mixer.music.get_pos() == -1:  # Если трек не начат
                    pygame.mixer.music.play(start=self.play_position)
                    self.play_start_time = time.time()
                    self.position_buffer.clear()
                    logging.info(f"Воспроизведение начато с {self.play_position:.2f} сек")
                else:
                    pygame.mixer.music.unpause()
                    self.play_start_time = time.time() - self.play_position
                    logging.info("Воспроизведение возобновлено")
                self.is_playing = True
                self.timer.start(50)
            except pygame.error as e:
                logging.error(f"Ошибка воспроизведения: {e}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось воспроизвести аудио: {e}")
                return

    def stop_playback(self):
        if self.audio_file:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.timer.stop()
            self.play_position = 0
            self.position_buffer.clear()
            if self.spectrogram is not None and self.spectrogram_image is not None:
                self.update_spectrogram_display()
            self.seek_slider.setValue(0)
            self.time_label.setText(f"00:00 / {format_time(self.total_duration)}")
            logging.info("Воспроизведение остановлено")

    def seek_position(self, value):
        """
        Перематывает трек на позицию, указанную слайдером.
        """
        if self.audio_file and self.total_duration > 0:
            # Переводим значение слайдера (0–1000) в секунды
            new_position = (value / 1000.0) * self.total_duration
            self.play_position = new_position
            self.position_buffer.clear()
            self.position_buffer.append(new_position)
            
            # Перематываем аудио
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.play(start=new_position)
                if not self.is_playing:
                    pygame.mixer.music.pause()  # Оставляем в паузе, если не воспроизводится
                self.play_start_time = time.time() - new_position
                logging.info(f"Перемотка на позицию: {new_position:.2f} сек (слайдер={value})")
                
                # Обновляем спектрограмму и прокрутку
                self.update_spectrogram_display()
                self.update_play_position()
            except pygame.error as e:
                logging.error(f"Ошибка при перемотке: {e}")
                QMessageBox.critical(self, "Ошибка", f"Не удалось перемотать аудио: {e}")

    def zoom_in(self):
        if self.spectrogram_image:
            self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)
            self.update_spectrogram_display()
            logging.info(f"Zoom In: масштаб={self.zoom_factor:.2f}")

    def zoom_out(self):
        if self.spectrogram_image and self.zoom_factor > 0.5:
            self.zoom_factor /= 1.2
            self.update_spectrogram_display()
            logging.info(f"Zoom Out: масштаб={self.zoom_factor:.2f}")

    def update_spectrogram_display(self):
        """
        Обновляет отображение спектрограммы с учётом текущего масштаба.
        """
        if self.spectrogram_image:
            scaled_pixmap = QPixmap.fromImage(self.spectrogram_image).scaled(
                int(self.image_width * self.zoom_factor),
                int(self.spectrogram_image.height() * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.spectrogram_label.setPixmap(scaled_pixmap)
            self.spectrogram_label.adjustSize()
            logging.info(f"Обновлено изображение: ширина={scaled_pixmap.width()}, высота={scaled_pixmap.height()}")

    def update_play_position(self):
        if self.is_playing and self.spectrogram is not None and self.spectrogram_image is not None:
            # Обновляем позицию на основе системного времени
            if pygame.mixer.music.get_busy():
                self.play_position = time.time() - self.play_start_time
                if self.play_position > self.total_duration:
                    self.play_position = self.total_duration
                    self.stop_playback()
                    return
                self.position_buffer.append(self.play_position)
                self.play_position = sum(self.position_buffer) / len(self.position_buffer)
            else:
                logging.warning("Воспроизведение завершено")
                self.stop_playback()
                return

            # Обновляем индикатор времени
            self.time_label.setText(f"{format_time(self.play_position)} / {format_time(self.total_duration)}")

            # Обновляем слайдер
            if self.total_duration > 0:
                slider_value = int((self.play_position / self.total_duration) * 1000)
                self.seek_slider.setValue(slider_value)

            # Логируем раз в ~1 сек
            self.log_counter += 1
            if self.log_counter % 20 == 0:
                logging.info(f"Позиция воспроизведения: {self.play_position:.2f} сек")

            # Создаём копию изображения с учётом масштаба
            scaled_pixmap = QPixmap.fromImage(self.spectrogram_image).scaled(
                int(self.image_width * self.zoom_factor),
                int(self.spectrogram_image.height() * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter = QPainter(scaled_pixmap)
            pen = QPen(QColor(Qt.white), 4, Qt.DashLine)
            painter.setPen(pen)
            max_time = self.time[-1]
            if max_time > 0 and self.data_area is not None:
                # Масштабируем x_pos на область данных с учётом зума
                data_width_pixels = scaled_pixmap.width() * self.data_area['width']
                data_x0_pixels = scaled_pixmap.width() * self.data_area['x0']
                x_pos = int((self.play_position / max_time) * data_width_pixels + data_x0_pixels)
                x_pos = min(max(x_pos, 0), scaled_pixmap.width() - 1)
                if self.log_counter % 20 == 0:
                    logging.info(f"Позиция линии: {x_pos} пикселей (x0_pixels={data_x0_pixels:.1f})")
                painter.drawLine(x_pos, 0, x_pos, scaled_pixmap.height())
            painter.end()

            # Обновляем изображение
            self.spectrogram_label.setPixmap(scaled_pixmap)

            # Прокручиваем область, чтобы линия была в центре
            if self.time is not None and max_time > 0:
                scroll_range = self.spectrogram_label.pixmap().width() - self.scroll_area.width()
                if scroll_range > 0:
                    scroll_pos = int(x_pos - self.scroll_area.width() / 2)
                    scroll_pos = min(max(scroll_pos, 0), scroll_range)
                    self.scroll_area.horizontalScrollBar().setValue(scroll_pos)
                    if self.log_counter % 20 == 0:
                        logging.info(f"Позиция прокрутки: {scroll_pos} пикселей")

    def save_spectrogram(self):
        if self.audio_file is None or self.spectrogram_image is None:
            QMessageBox.critical(self, "Ошибка", "Сначала загрузите аудиофайл!")
            return

        output_path = os.path.splitext(self.audio_file)[0] + "_fft_spectrogram.png"
        self.spectrogram_image.save(output_path)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        QMessageBox.information(
            self, "Сохранено", 
            f"Спектрограмма сохранена как {output_path}\nРазмер: {file_size_mb:.2f} MB"
        )

    def closeEvent(self, event):
        pygame.mixer.quit()
        event.accept()

def main():
    app = QApplication([])
    window = SpectrogramWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()

