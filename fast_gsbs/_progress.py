"""Stand-in for the part of tqdm that the reference fit() uses. fit(showProgressBar=True) prints
a one-line counter to stderr."""
import sys


class progress:
    def __init__(self, total=None, disable=False):
        self.total = total
        self.n = 0
        self.disable = disable

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.disable:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return False

    def update(self, n=1):
        self.n += n
        if not self.disable:
            sys.stderr.write("\rGSBS: %d/%s" % (self.n, self.total))
            sys.stderr.flush()
