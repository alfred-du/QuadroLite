from __future__ import annotations

import logging
import pathlib

import psutil

logger = logging.getLogger(__name__)

_THERMAL_ZONE = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")


def memory_usage_mb() -> float:
    """Return current process RSS in megabytes."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def cpu_temp_c() -> float | None:
    """Read SoC temperature on Raspberry Pi. Returns None on non-RPi hosts."""
    try:
        return int(_THERMAL_ZONE.read_text().strip()) / 1000.0
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def log_health(fps: float) -> None:
    mem = memory_usage_mb()
    temp = cpu_temp_c()
    temp_str = f"{temp:.1f}°C" if temp is not None else "n/a"
    logger.info("fps=%.1f  mem=%.0fMB  temp=%s", fps, mem, temp_str)
