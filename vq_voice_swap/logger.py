class Logger:
    """
    Log training iterations to a file and to standard output.
    """

    def __init__(self, out_filename: str, resume: bool = False):
        if resume:
            self.out_file = open(out_filename, "a+")
            # TODO: get the latest step in the log file and add it
            # to future log messages when resuming.
        else:
            self.out_file = open(out_filename, "w+")

    def log(self, step: int, **kwargs):
        fields = " ".join(f"{k}={v:.05f}" for k, v in kwargs.items())
        print(f"step {step}: {fields}")
        self.out_file.write(f"step {step}: {fields}\n")
        self.out_file.flush()

    def close(self):
        self.out_file.close()
