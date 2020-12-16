import json
import os
import subprocess
from typing import Dict, Optional, Union

import numpy as np
from torch.utils.data import Dataset

DURATION_ESTIMATE_SLACK = 0.05


class LibriSpeech(Dataset):
    def __init__(
        self,
        directory: str,
        window_duration: float = 4.0,
        window_spacing: float = 0.05,
        sample_rate: int = 16000,
    ):
        self.directory = directory
        self.window_duration = window_duration
        self.window_spacing = window_spacing
        self.sample_rate = sample_rate

        index_path = os.path.join(self.directory)
        if os.path.exists(index_path):
            with open(index_path, "rt") as f:
                self.index = json.load(f)
        else:
            self.index = _build_file_index(directory)
            with open(index_path, "wt") as f:
                json.dump(self.index, f)

        self.speaker_ids = sorted(self.index.keys())
        self.data = []
        for label, speaker_id in enumerate(self.speaker_ids):
            self._create_speaker_data(
                label, os.path.join(self.directory, speaker_id), self.index[speaker_id]
            )

    def _create_speaker_data(
        self, label: int, path: str, index_dict: Dict[str, Union[Dict, float]]
    ):
        for name, item in index_dict.items():
            sub_path = os.path.join(path, name)
            if isinstance(item, float):
                window_samples = int(self.sample_rate * self.window_duration)
                space_samples = int(self.sample_rate * self.window_spacing)
                total_samples = int(self.sample_rate * (item - DURATION_ESTIMATE_SLACK))
                idx = 0
                while idx + window_samples < total_samples:
                    self.data.append(LibriSpeechDatum(label, sub_path, 0))
                    idx += space_samples
            else:
                self._create_speaker_data(label, sub_path, item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Union[int, np.ndarray]]:
        datum = self.data[index]
        reader = ChunkReader(datum.path, self.sample_rate)
        try:
            reader.read(datum.offset)
            num_samples = int(self.sample_rate * self.window_duration)
            samples = reader.read(num_samples)
            assert len(samples) == num_samples, "file was shorter than expected"
            return {"label": datum.label, "samples": samples}
        finally:
            reader.close()


class LibriSpeechDataError(Exception):
    pass


class LibriSpeechDatum:
    def __init__(self, label: int, path: str, offset: int):
        self.label = label
        self.path = path
        self.offset = offset


def _build_file_index(data_dir: str) -> Dict[str, Union[Dict, float]]:
    result = {}
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if item.endswith(".flac") and not item.startswith("."):
            result[item] = _lookup_audio_duration(item_path)
        else:
            sub_result = _build_file_index(item_path)
            if len(sub_result):
                result[item] = sub_result
    return result


class ChunkReader:
    """
    An API for reading chunks of audio samples from an audio file.

    :param path: the path to the audio file.
    :param sample_rate: the number of samples per second, used for resampling.

    Adapted from https://github.com/unixpickle/udt-voice-swap/blob/9ab0404c3e102ec19709c2d6e9763ae629b4f897/voice_swap/data.py#L63
    """

    def __init__(self, path: str, sample_rate: int):
        self.path = path
        self.sample_rate = sample_rate
        self._done = False

        audio_reader, audio_writer = os.pipe()
        try:
            args = [
                "ffmpeg",
                "-i",
                path,
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "pipe:%i" % audio_writer,
            ]
            self._ffmpeg_proc = subprocess.Popen(
                args,
                pass_fds=(audio_writer,),
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            self._audio_reader = audio_reader
            audio_reader = None
        finally:
            os.close(audio_writer)
            if audio_reader is not None:
                os.close(audio_reader)

        self._reader = os.fdopen(self._audio_reader, "rb")

    def read(self, chunk_size: int) -> Optional[np.ndarray]:
        """
        Read a chunk of audio samples from the file.

        :param chunk_size: the number of samples to read.
        :return: A chunk of audio, represented as a 1-D numpy array of floats,
                 where each sample is in the range [-1, 1].
                 When there are no more samples left, None is returned.
        """
        buf = self.read_raw(chunk_size)
        if buf is None:
            return None
        return np.frombuffer(buf, dtype="int16").astype("float32") / (2 ** 15)

    def read_raw(self, chunk_size) -> Optional[bytes]:
        if self._done:
            return None
        buffer_size = chunk_size * 2
        buf = self._reader.read(buffer_size)
        if len(buf) < buffer_size:
            self._done = True
        if not len(buf):
            return None
        return buf

    def close(self):
        if not self._done:
            self._reader.close()
            self._ffmpeg_proc.wait()
        else:
            self._ffmpeg_proc.wait()
            self._reader.close()


def _lookup_audio_duration(path: str) -> float:
    p = subprocess.Popen(
        ["ffmpeg", "-i", path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, output = p.communicate()
    output = str(output, "utf-8")
    lines = [x.strip() for x in output.split("\n")]
    duration_lines = [x for x in lines if x.startswith("Duration:")]
    if len(duration_lines) != 1:
        raise ValueError(f"unexpected output from ffmpeg for: {path}")
    duration_str = duration_lines[0].split(" ")[1].split(",")[0]
    hours, minutes, seconds = [float(x) for x in duration_str.split(":")]
    return seconds + (minutes + hours * 60) * 60
