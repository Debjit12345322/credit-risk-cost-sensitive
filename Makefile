# Every stage of the project, in order. `make all` runs the full pipeline.
.PHONY: download data audit analysis quick test check all clean

download:  ## fetch the raw Kaggle file (needs ~/.kaggle/kaggle.json)
	kaggle datasets download -d wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/

data:      ## raw file -> labelled 2014 parquet + censoring census
	python scripts/build_dataset.py

audit:     ## 152-column leakage audit + naive-vs-audited demo
	python scripts/run_leakage_audit.py

analysis:  ## every README number, five seeds
	python scripts/run_full_analysis.py

quick:     ## primary seed only
	python scripts/run_full_analysis.py --quick

test:      ## unit tests (synthetic data, no download needed)
	python -m pytest -q

check:     ## every number in README.md must exist in reports/
	python scripts/check_readme_numbers.py

all: data audit analysis test check

clean:     ## remove generated artefacts (keeps the raw download)
	rm -f data/*.parquet reports/*.md reports/*.json reports/figures/*.png
