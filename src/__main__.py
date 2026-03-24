"""Allow ``python -m src``."""
from src.main import _configure_logging, main

_configure_logging()
main()
