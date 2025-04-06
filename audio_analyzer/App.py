import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import csv
import matplotlib.patches as mpatches
import sounddevice as sd

def autocorrelation(s):
    N = len(s)
    return np.array([np.sum(s[i:] * s[:N-i]) for i in range(N)])

def amdf(s):
    N = len(s)
    return np.array([np.sum(np.abs(s[i:] - s[:N - i])) for i in range(N)])

def zero_crossing_rate(s):
    return np.array([((s[:-1] * s[1:]) < 0).sum() / len(s)])

def volume(s):
    return np.array([np.sqrt(np.mean(s**2))])

def ste(s):
    return np.array([np.sum(s**2)])

class LeftPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, width=200)
        self.grid(row=0, column=0, rowspan=3, sticky="ns", padx=5, pady=10)
        self.tiles = {}

    def add_tile(self, filepath, on_export, on_remove):
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", pady=3)
        label = ctk.CTkLabel(frame, text=os.path.basename(filepath), anchor="w")
        label.pack(side="left", padx=2)
        export_btn = ctk.CTkButton(frame, text="💾", width=10, command=on_export)
        export_btn.pack(side="right", padx=2)
        remove_btn = ctk.CTkButton(frame, text="🗑", width=10, command=on_remove)
        remove_btn.pack(side="right", padx=2)
        self.tiles[filepath] = frame

    def remove_tile(self, filepath):
        if filepath in self.tiles:
            self.tiles[filepath].destroy()
            del self.tiles[filepath]

class RightPanel(ctk.CTkFrame):
    def __init__(self, master, on_update_callback):
        super().__init__(master, width=150)
        self.on_update_callback = on_update_callback
        self.grid(row=0, column=2, rowspan=3, sticky="ns", padx=5, pady=10)
        self.silence_cb = ctk.CTkCheckBox(self, text="Silence", command=on_update_callback)
        self.voiced_cb = ctk.CTkCheckBox(self, text="Voiced", command=on_update_callback)
        self.unvoiced_cb = ctk.CTkCheckBox(self, text="Unvoiced", command=on_update_callback)
        self.music_cb = ctk.CTkCheckBox(self, text="Music", command=on_update_callback)
        self.speech_cb = ctk.CTkCheckBox(self, text="Speech", command=on_update_callback)
        for cb in [self.silence_cb, self.voiced_cb, self.unvoiced_cb, self.music_cb, self.speech_cb]:
            cb.pack(anchor="w")

    def get_filters(self):
        return {
            "silence": self.silence_cb.get(),
            "voiced": self.voiced_cb.get(),
            "unvoiced": self.unvoiced_cb.get(),
            "music": self.music_cb.get(),
            "speech": self.speech_cb.get()
        }

class TopPanel(ctk.CTkFrame):
    def __init__(self, master, on_update_callback):
        super().__init__(master)
        self.on_update_callback = on_update_callback
        self.grid(row=0, column=1, sticky="nsew", padx=20, pady=(10, 0))

        self.load_button = ctk.CTkButton(self, text="Load Audio Files", command=self.on_update_callback)
        self.columnconfigure((0, 1, 2), weight=1)
        self.load_button.grid(row=0, column=0, padx=10, sticky="w")
        self.frame_length_frame = ctk.CTkFrame(self)
        self.frame_length_frame.grid(row=0, column=1, padx=10, sticky="ew")
        self.frame_length_slider = ctk.CTkSlider(self.frame_length_frame, from_=10, to=16384, number_of_steps=160, command=self.frame_length_slider_changed)
        self.frame_length_slider.set(2048)
        self.frame_length_frame.columnconfigure(0, weight=1)
        self.frame_length_slider.grid(row=0, column=0, sticky="ew")
        self.frame_length_label = ctk.CTkLabel(self.frame_length_frame, text=f"Frame length: {self.get_frame_length()}")
        self.frame_length_label.grid(row=1, column=0)

        self.frame_step_frame = ctk.CTkFrame(self)
        self.frame_step_frame.grid(row=0, column=2, padx=10, sticky="ew")
        self.frame_step_slider = ctk.CTkSlider(self.frame_step_frame, from_=10, to=8192, number_of_steps=160, command=self.frame_step_slider_changed)
        self.frame_step_slider.set(512)
        self.frame_step_slider.pack()
        self.frame_step_label = ctk.CTkLabel(self.frame_step_frame, text=f"Frame step: {self.get_frame_step()}")
        self.frame_step_label.pack()

        self.checkbox_frame = ctk.CTkFrame(self)
        self.checkbox_frame.grid(row=1, column=1, sticky="ew", padx=20, pady=(5, 10))
        def add_checkbox(name):
            cb = ctk.CTkCheckBox(self.checkbox_frame, text=name, command=self.on_update_callback)
            cb.pack(side="left", padx=5)
            return cb
        self.show_waveform_checkbox = add_checkbox("Waveform")
        self.show_waveform_checkbox.select()
        self.show_spectrogram_checkbox = add_checkbox("Spectrogram")
        self.show_spectrogram_checkbox.select()
        self.show_autocorrelation_checkbox = add_checkbox("Auto-cor")
        self.show_amdf_checkbox = add_checkbox("AMDF")
        self.show_zcr_checkbox = add_checkbox("ZCR")
        self.show_volume_checkbox = add_checkbox("Volume")
        self.show_ste_checkbox = add_checkbox("STE")

    def show_waveform(self): return self.show_waveform_checkbox.get()
    def show_spectrogram(self): return self.show_spectrogram_checkbox.get()
    def show_autocorrelation(self): return self.show_autocorrelation_checkbox.get()
    def show_amdf(self): return self.show_amdf_checkbox.get()
    def show_zcr(self): return self.show_zcr_checkbox.get()
    def show_volume(self): return self.show_volume_checkbox.get()
    def show_ste(self): return self.show_ste_checkbox.get()
    def get_frame_length(self): return int(self.frame_length_slider.get())
    def get_frame_step(self): return int(self.frame_step_slider.get())
    def frame_length_slider_changed(self, value):
        self.on_update_callback()
        self.frame_length_label.configure(text=f"Frame length: {self.get_frame_length()}")
    def frame_step_slider_changed(self, value):
        self.on_update_callback()
        self.frame_step_label.configure(text=f"Frame step: {self.get_frame_step()}")

class AudioPanel:
    def __init__(self, parent, filepath, y, sr):
        self.parent = parent
        self.filepath = filepath
        self.y = y
        self.sr = sr
        self.playback_pos = 0
        self.is_playing = False
        self.panel = ctk.CTkFrame(parent.scrollable_frame)
        self.panel.pack(pady=10, padx=10, fill="x")
        name_frame = ctk.CTkFrame(self.panel)
        name_frame.pack(pady=5, fill="x")
        self.name_label = ctk.CTkLabel(name_frame, text=os.path.basename(self.filepath),
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.name_label.pack(side="left", padx=5)
        btn_frame = ctk.CTkFrame(name_frame)
        btn_frame.pack(side="right", padx=5)
        play_btn = ctk.CTkButton(btn_frame, text="▶️", width=30, command=self.play_audio)
        play_btn.pack(side="left", padx=2)
        stop_btn = ctk.CTkButton(btn_frame, text="⏹️", width=30, command=self.pause_audio)
        stop_btn.pack(side="left", padx=2)

        self.plot_areas = []
        self.update_contents()

    def remove(self):
        self.panel.destroy()
        self.parent.left_panel.remove_tile(self.filepath)
        self.parent.audio_panels.remove(self)

    def update_contents(self):
        for widget in self.panel.winfo_children():
            if isinstance(widget, ctk.CTkCanvas):
                widget.destroy()

        y, sr = self.y, self.sr

        if self.parent.top_panel.show_waveform():
            def plot_waveform_with_overlay(ax):
                librosa.display.waveshow(y, sr=sr, ax=ax)
                filters = self.parent.right_panel.get_filters()
                frame_length = self.parent.top_panel.get_frame_length()
                frame_step = self.parent.top_panel.get_frame_step()
                volume_vals, zcr_vals, time_stamps = [], [], []
                for start in range(0, len(y) - frame_length + 1, frame_step):
                    frame = y[start:start + frame_length]
                    volume_vals.append(volume(frame)[0])
                    zcr_vals.append(zero_crossing_rate(frame)[0])
                    time_stamps.append(start / sr)
                vol_thresh = np.percentile(volume_vals, 25)
                zcr_thresh = np.percentile(zcr_vals, 25)
                zcr_high = np.percentile(zcr_vals, 75)
                for t, v, z in zip(time_stamps, volume_vals, zcr_vals):
                    if filters["silence"] and v < vol_thresh and z < zcr_thresh:
                        ax.axvspan(t, t + frame_step / sr, color='blue', alpha=0.3)
                    if filters["unvoiced"] and v < vol_thresh and z > zcr_high:
                        ax.axvspan(t, t + frame_step / sr, color='green', alpha=0.3)
                    if filters["voiced"] and v > vol_thresh and z < zcr_high:
                        ax.axvspan(t, t + frame_step / sr, color='red', alpha=0.3)
                    if filters["music"] and v > vol_thresh and z > zcr_high:
                        ax.axvspan(t, t + frame_step / sr, color='yellow', alpha=0.3)
                    if filters["speech"] and v > vol_thresh and z < zcr_high:
                        ax.axvspan(t, t + frame_step / sr, color='purple', alpha=0.3)
                legend_patches = []
                if filters["silence"]:
                    legend_patches.append(mpatches.Patch(color='blue', alpha=0.3, label='Silence'))
                if filters["unvoiced"]:
                    legend_patches.append(mpatches.Patch(color='green', alpha=0.3, label='Unvoiced'))
                if filters["voiced"]:
                    legend_patches.append(mpatches.Patch(color='red', alpha=0.3, label='Voiced'))
                if filters["music"]:
                    legend_patches.append(mpatches.Patch(color='yellow', alpha=0.3, label='Music'))
                if filters["speech"]:
                    legend_patches.append(mpatches.Patch(color='purple', alpha=0.3, label='Speech'))

                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper right')

            self.display_plot(plot_waveform_with_overlay, "Waveform")

        if self.parent.top_panel.show_spectrogram():
            S = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            fig, ax = plt.subplots(figsize=(8, 2))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
            ax.set_title("Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).resize((800, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.spectrogram_image = photo
            self.spectrogram_canvas = ctk.CTkCanvas(self.panel, width=800, height=200)
            self.spectrogram_canvas.create_image(0, 0, anchor='nw', image=photo)
            self.spectrogram_canvas.image = photo
            self.spectrogram_canvas.pack(pady=5)

        for show_flag, func, title in [
            (self.parent.top_panel.show_autocorrelation(), autocorrelation, "Autocorrelation"),
            (self.parent.top_panel.show_amdf(), amdf, "AMDF"),
            (self.parent.top_panel.show_zcr(), zero_crossing_rate, "Zero Crossing Rate"),
            (self.parent.top_panel.show_volume(), volume, "Volume"),
            (self.parent.top_panel.show_ste(), ste, "Short-Time Energy")
        ]:
            if show_flag:
                values = self.apply_function(func)
                self.display_plot(lambda ax: ax.plot(values), title)
                if title == "Volume":
                    self.display_volume_stats(values)

    def apply_function(self, func):
        frame_length = self.parent.top_panel.get_frame_length()
        frame_step = self.parent.top_panel.get_frame_step()
        results = []
        for start in range(0, len(self.y) - frame_length + 1, frame_step):
            frame = self.y[start:start + frame_length]
            result = func(frame)
            results.append(np.mean(result))
        return np.array(results)

    def display_plot(self, plot_func, title=""):
        fig, ax = plt.subplots(figsize=(8, 2))
        plot_func(ax)
        ax.set_title(title)
        if "Volume" in title:
            ax.set_ylabel("Volume")
        elif "Zero Crossing Rate" in title:
            ax.set_ylabel("Zero Crossing Rate")
        elif "Short-Time Energy" in title:
            ax.set_ylabel("Short Time Energy")
        elif "AMDF" in title:
            ax.set_ylabel("AMDF")
        elif "Autocorrelation" in title:
            ax.set_ylabel("Autocorrelation")
        else:
            ax.set_ylabel("Value")
        ax.set_xlabel("Frame Index")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).resize((800, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        canvas = ctk.CTkCanvas(self.panel, width=800, height=200)
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.image = photo
        canvas.pack(pady=5)

    def display_volume_stats(self, values):
        vstd = np.std(values)
        vdr = np.max(values) - np.min(values)
        vund = np.mean(np.abs(np.diff(values)))
        stats = f"Volume STD: {vstd:.4f} | Dynamic Range: {vdr:.4f} | Undulation: {vund:.4f}"
        ctk.CTkLabel(self.panel, text=stats).pack(pady=5)

    def export_data(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return
        fl = self.parent.top_panel.get_frame_length()
        fs = self.parent.top_panel.get_frame_step()
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Start", "Volume", "STE", "ZCR", "Autocorr", "AMDF"])
            for i, start in enumerate(range(0, len(self.y) - fl + 1, fs)):
                frame = self.y[start:start + fl]
                writer.writerow([
                    i, start,
                    volume(frame)[0],
                    ste(frame)[0],
                    zero_crossing_rate(frame)[0],
                    np.mean(autocorrelation(frame)),
                    np.mean(amdf(frame))
                ])

    def play_audio(self):
        try:
            if self.is_playing:
                return
            self.is_playing = True
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                callback=self.audio_callback,
                finished_callback=self.on_stream_finished
            )
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Unable to play", e)

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        end = self.playback_pos + frames
        chunk = self.y[self.playback_pos:end]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):] = 0
            raise sd.CallbackStop()
        else:
            outdata[:, 0] = chunk
        self.playback_pos = end

    def pause_audio(self):
        if self.is_playing and hasattr(self, 'stream'):
            self.stream.stop()
            self.is_playing = False

    def on_stream_finished(self):
        self.is_playing = False
        self.playback_pos = 0

class AudioViewer(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.left_panel = LeftPanel(self)
        self.left_panel.grid(row=0, column=0, rowspan=3, sticky="ns", padx=5, pady=10)

        self.right_panel = RightPanel(self, self.update_all_panels)
        self.right_panel.grid(row=0, column=2, rowspan=3, sticky="ns", padx=5, pady=10)

        self.top_panel = TopPanel(self, self.update_all_panels)
        self.top_panel.grid(row=0, column=1, sticky="ew")

        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=850, height=900)
        self.scrollable_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=10)
        self.audio_panels = []

    def load_audio_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")])
        if not filepaths:
            return
        for filepath in filepaths:
            try:
                y, sr = librosa.load(filepath)
                panel = AudioPanel(self, filepath, y, sr)
                self.audio_panels.append(panel)
                self.left_panel.add_tile(filepath, panel.export_data, panel.remove)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {os.path.basename(filepath)}: {e}")

    def update_all_panels(self, *_):
        for panel in self.audio_panels:
            panel.update_contents()

class AudioAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("Audio Analyzer")
        self.geometry("1800x1000")
        self.audio_viewer = AudioViewer(self)
        self.audio_viewer.pack(expand=True, fill="both")
        self.audio_viewer.top_panel.load_button.configure(command=self.audio_viewer.load_audio_files)

if __name__ == '__main__':
    app = AudioAnalyzerApp()
    app.mainloop()