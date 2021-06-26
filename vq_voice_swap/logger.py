from typing import Any, Dict, Iterator, TextIO, Tuple, Union

# The log line indicating that a checkpoint was saved.
SAVED_MSG = "# saved\n"


def read_log(log_reader: Union[str, TextIO]) -> Iterator[Tuple[int, Dict[str, Any]]]:
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
        elif line.startswith("#"):
            continue
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

    The log includes a dict of keys and numerical values for each step, as
    well as optional markers whenever checkpoints were saved to a file.

    The log can be resumed, in which case it is automatically truncated to the
    last save (or not truncated, if no saves are marked).
    To access the step of the first log message from a resume, look at the
    start_step attribute.
    """

    def __init__(self, out_filename: str, resume: bool = False):
        self.start_step = 0
        if resume:
            with open(out_filename, "r") as in_file:
                all_lines = in_file.readlines()

            # The log may not include a save due to legacy code, but if
            # it does, we should truncate to it.
            if SAVED_MSG in all_lines:
                keep_lines = len(all_lines) - all_lines[::-1].index(SAVED_MSG)
                all_lines = all_lines[:keep_lines]

            step_lines = [x for x in all_lines if x.startswith("step ")]
            if len(step_lines):
                self.start_step = int(step_lines[-1].split(" ")[1].split(":")[0])

            # Re-write the (possibly truncated) log.
            self.out_file = open(out_filename, "w+")
            self.out_file.write("".join(all_lines))
            self.out_file.flush()
        else:
            self.out_file = open(out_filename, "w+")

    def log(self, step: int, **kwargs):
        fields = " ".join(f"{k}={v:.05f}" for k, v in kwargs.items())
        log_line = f"step {step + self.start_step}: {fields}"
        self.out_file.write(log_line + "\n")
        self.out_file.flush()
        print(log_line)

    def mark_save(self):
        self.out_file.write(SAVED_MSG)
        self.out_file.flush()

    def close(self):
        self.out_file.close()
