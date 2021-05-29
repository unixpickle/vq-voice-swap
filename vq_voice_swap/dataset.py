import json
import os
import subprocess
from typing import Dict, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

DURATION_ESTIMATE_SLACK = 0.05


def create_data_loader(
    directory: str, batch_size: int, num_workers=4, **dataset_kwargs
) -> Tuple[DataLoader, int]:
    """
    Create an audio data loader, either from LibriSpeech or from a small
    synthetic set of tones.

    Returned batches are dicts containing at least two keys:
        - 'label' (int): the speaker ID.
        - 'samples' (tensor): an [N x T] batch of samples.

    :param directory: the LibriSpeech data directory, or "tones" to use a
                      placeholder dataset.
    :param batch_size: the number of samples per batch.
    :param num_workers: number of parallel data loading threads.
    :return: a pair (loader, num_labels), where loader is the DataLoader and
             num_labels is one greater than the maximum label index.
    """
    if directory == "tones":
        dataset = ToneDataset()
    else:
        dataset = LibriSpeech(directory, **dataset_kwargs)
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),
        len(dataset.speaker_ids),
    )


class LibriSpeech(Dataset):
    def __init__(
        self,
        directory: str,
        window_duration: float = 4.0,
        window_spacing: float = 0.2,
        sample_rate: int = 16000,
    ):
        self.directory = directory
        self.window_duration = window_duration
        self.window_spacing = window_spacing
        self.sample_rate = sample_rate

        index_path = os.path.join(self.directory, "index.json")
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
                if window_samples >= total_samples:
                    self.data.append(LibriSpeechDatum(label, sub_path, 0))
                else:
                    while idx + window_samples < total_samples:
                        self.data.append(LibriSpeechDatum(label, sub_path, idx))
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
            samples = np.pad(samples, (0, num_samples - len(samples)))
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


class ToneDataset(Dataset):
    """
    A dataset where each "speaker" is a different frequency and each sample is
    just a phase-shifted sinusoidal wave.
    """

    def __init__(self):
        self.speaker_ids = [300, 500, 1000]

    def __len__(self):
        return len(self.speaker_ids) * 10

    def __getitem__(self, index) -> Dict[str, Union[int, np.ndarray]]:
        speaker = index % len(self.speaker_ids)
        frequency = self.speaker_ids[speaker]
        phase = (index // len(self.speaker_ids)) / 10

        data = np.arange(0, 64000, step=1).astype(np.float32) / 16000
        coeffs = (data + phase) * np.pi * 2 * frequency
        return {
            "label": speaker,
            "samples": np.sin(coeffs),
        }


def _build_file_index(data_dir: str) -> Dict[str, Union[Dict, float]]:
    result = {}
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if item.endswith(".flac") and not item.startswith("."):
            result[item] = lookup_audio_duration(item_path)
        elif os.path.isdir(item_path):
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


class ChunkWriter:
    """
    An API for writing chunks of audio samples from an audio file.

    :param path: the path to the audio file.
    :param sample_rate: the number of samples per second to write.
    """

    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate

        audio_reader, audio_writer = os.pipe()
        try:
            audio_format = ["-ar", str(sample_rate), "-ac", "1", "-f", "s16le"]
            audio_params = audio_format + [
                "-probesize",
                "32",
                "-thread_queue_size",
                "60",
                "-i",
                "pipe:%i" % audio_reader,
            ]
            output_params = [path]
            self._ffmpeg_proc = subprocess.Popen(
                ["ffmpeg", "-y", *audio_params, *output_params],
                pass_fds=(audio_reader,),
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            self._audio_writer = audio_writer
            audio_writer = None
        finally:
            if audio_writer is not None:
                os.close(audio_writer)
            os.close(audio_reader)

        self._writer = os.fdopen(self._audio_writer, "wb", buffering=1024)

    def write(self, chunk):
        """
        Read a chunk of audio samples from the file.

        :param chunk: a chunk of samples, stored as a 1-D numpy array of floats,
                      where each sample is in the range [-1, 1].
        """
        chunk = np.clip(chunk, -1, 1)
        data = bytes((chunk * (2 ** 15 - 1)).astype("int16"))
        self._writer.write(data)

    def close(self):
        self._writer.close()
        self._ffmpeg_proc.wait()


def lookup_audio_duration(path: str) -> float:
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
