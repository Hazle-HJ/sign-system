#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python src/remeasure.py
python src/dataset.py
python src/dna.py
python src/report.py
open data/report.html
