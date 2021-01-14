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
