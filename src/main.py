"""Aegra entry point, referenced in aegra.json as ./src/main.py:graph."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()
logging.basicConfig(level=logging.INFO)

from coach.graph import create_graph  # noqa: E402

graph = create_graph()
