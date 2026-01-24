from __future__ import annotations
from __future__ import print_function

import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm


def tqdm_sink(msg: str) -> None:
    tqdm.write(msg.rstrip("\n"))


@dataclass
class Logger:
    app: str = "PROJECT"
    level: str = "INFO"
    log_dir: str | Path = "logs"
    use_tqdm_sink: bool = True
    rich_tracebacks: bool = True
    serialize: bool = False

    def __post_init__(self) -> None:
        self.run_id = uuid.uuid4().hex[:8].upper()
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        day = datetime.now().strftime("%Y-%m-%d")
        self.main_file = self.log_dir / f"{self.app.lower()}_{day}.log"
        self.err_file = self.log_dir / f"{self.app.lower()}_{day}_errors.log"

        # keep format simple and consistent:
        # time | level | app:run | file:line func | message
        self.fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            f"<cyan>{self.app}</cyan>:<magenta>{self.run_id}</magenta> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> <cyan>{function}</cyan> - "
            "<level>{message}</level>"
        )

        self._t0: Optional[float] = None
        self.setup()

    @property
    def log(self):
        return self._log

    def setup(self) -> None:
        logger.remove()

        sink = tqdm_sink if self.use_tqdm_sink else sys.stdout
        logger.add(
            sink, level=self.level, colorize=True,
            format=self.fmt, enqueue=True, 
            backtrace=self.rich_tracebacks, diagnose=False)

        logger.add(
            self.main_file, level="DEBUG",
            format=self.fmt, enqueue=True,
            compression="zip", serialize=self.serialize)

        logger.add(
            self.err_file, level="ERROR",
            format=self.fmt, enqueue=True,
            compression="zip", serialize=self.serialize)

        self._log = logger.bind(app=self.app, run_id=self.run_id)

    def debug(self, *a, **k): return self._log.debug(*a, **k)
    def info(self, *a, **k): return self._log.info(*a, **k)
    def warning(self, *a, **k): return self._log.warning(*a, **k)
    def error(self, *a, **k): return self._log.error(*a, **k)

    def set_level(self, level: str) -> None:
        self.level = level
        self.setup()

    def section(self, title: str, level: str = "INFO", char: str = "=") -> None:
        line = char * max(8, len(title) + 6)
        self._log.log(level, "")
        self._log.log(level, line)
        self._log.log(level, title)
        self._log.log(level, line)

    def rule(self, level: str = "INFO", char: str = "-", width: int = 56) -> None:
        self._log.log(level, char * width)

    def start_task(self, name: str, level: str = "INFO") -> None:
        self._t0 = time.perf_counter()
        self._log.log(level, f"{name} | start")

    def end_task(self, name: str, level: str = "SUCCESS") -> None:
        t1 = time.perf_counter()
        dt_ms = (t1 - (self._t0 or t1)) * 1000.0
        self._log.log(level, f"{name} | done in {dt_ms:.1f} ms")
        self._t0 = None

    def progress(self, total: int, desc: str = "", leave: bool = True) -> tqdm:
        return tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True, mininterval=0.2)
