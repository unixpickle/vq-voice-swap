from typing import TextIO, Union


def read_log(log_reader: Union[str, TextIO]):
    """
    Read entries in a log file as dicts.

    Returns an iterator over (step, dict) pairs.
    """
    if isinstance(log_reader, str):
        with open(log_reader, "rt") as f:
            yield from read_log(f)
            return
    line_idx = 0
    while True:
        line = log_reader.readline().rstrip()
        line_idx += 1
        if not line:
            break
        try:
            if not line.startswith("step "):
                raise ValueError
            step_str, kv_str = line[5:].split(": ")
            step_idx = int(step_str)
            kv_strs = kv_str.split(" ")
            kvs = {}
            for kv_str in kv_strs:
                k_str, v_str = kv_str.split("=")
                kvs[k_str] = float(v_str)
        except ValueError:
            raise ValueError(f"unexpected format at line {line_idx}")
        yield step_idx, kvs


class Logger:
    """
    Log training iterations to a file and to standard output.
    """

    def __init__(self, out_filename: str, resume: bool = False):
        self.start_step = 0
        if resume:
            self.out_file = open(out_filename, "a+")
            self.out_file.seek(0)
            lines = [x for x in self.out_file.readlines() if x.startswith("step ")]
            if len(lines):
                self.start_step = int(lines[-1].split(" ")[1].split(":")[0])
        else:
            self.out_file = open(out_filename, "w+")

    def log(self, step: int, **kwargs):
        fields = " ".join(f"{k}={v:.05f}" for k, v in kwargs.items())
        log_line = f"step {step + self.start_step}: {fields}"
        self.out_file.write(log_line)
        self.out_file.flush()
        print(log_line)

    def close(self):
        self.out_file.close()
