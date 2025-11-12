# vectorlite/lock.py
import os
import time
import errno
import contextlib
import sys

if os.name == "nt":
    import msvcrt
else:
    import fcntl

class FileLock:
    """
    Simple cross-platform advisory file lock (context manager).
    Uses fcntl on Unix and msvcrt on Windows.
    Lock file is created if missing. Blocking lock by default.
    """

    def __init__(self, lock_path: str, timeout: float = 10.0, poll_interval: float = 0.05):
        self.lock_path = os.path.abspath(lock_path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd = None

    def acquire(self):
        # Ensure directory exists
        d = os.path.dirname(self.lock_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        # Open or create lock file
        # Use r+ if exists else w+
        mode = "a+b" if os.name == "nt" else "a+"
        self._fd = open(self.lock_path, mode)
        start = time.time()
        while True:
            try:
                if os.name == "nt":
                    # msvcrt.locking on Windows - lock entire file
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Got it
                return
            except (BlockingIOError, OSError) as e:
                # On some systems BlockingIOError isn't raised, but errno EACCES/EAGAIN may be.
                if time.time() - start >= self.timeout:
                    raise TimeoutError(f"Timeout acquiring lock {self.lock_path}") from e
                time.sleep(self.poll_interval)

    def release(self):
        if not self._fd:
            return
        try:
            if os.name == "nt":
                try:
                    self._fd.seek(0)
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                try:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                self._fd.close()
            except Exception:
                pass
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
