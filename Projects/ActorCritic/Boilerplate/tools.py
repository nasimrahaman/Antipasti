"""Tools for training a actor critic pair."""

from collections import deque

class edb(object):
    def __init__(self, maxsize=100):
        """Basically a stack."""
        self.maxsize = maxsize

        raise NotImplementedError
        self.db = []

    def log(self, record):
        if len(self.db) >= self.maxsize:
            self.db.pop(0)
        self.db.append(record)

    def fetch(self):
        try:
            return self.db.pop()
        except IndexError:
            return None
