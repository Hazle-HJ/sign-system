#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python -u src/batch.py
python src/remeasure.py
python src/dataset.py
python src/dna.py
python src/report.py
open data/report.html
read -n 1 -s -r -p "끝났습니다. 아무 키나 누르면 이 창이 닫힙니다."
