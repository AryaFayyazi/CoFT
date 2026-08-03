# COFT -- reproduction entry points.
#
#   make setup      install the package and fetch the benchmarks
#   make test       fast unit tests (no model download)
#   make smoke      tiny end-to-end run of every table on one model
#   make all        the full main-text reproduction (Tables 1-4, Figs 3-4)
#
# Override the model with:  make table1 MODEL=configs/models/llama2-13b.yaml

PY      ?= python
MODEL   ?= configs/models/mistral-7b-instruct.yaml
MODELS  ?= configs/models/llama2-13b.yaml configs/models/mistral-7b-instruct.yaml
SMOKE   ?= configs/smoke.yaml
EXTRA   ?=

.PHONY: setup test smoke lint clean \
        table1 table2 table3 table4 sweeps tables figures all all-models

setup:
	$(PY) -m pip install -e ".[mauve]"
	$(PY) scripts/fetch_data.py

test:
	$(PY) -m pytest tests -q

lint:
	ruff check coft scripts tests

# --------------------------------------------------------------------------- #
# individual tables
# --------------------------------------------------------------------------- #
table1:
	$(PY) scripts/run_bias.py       --config $(MODEL) $(EXTRA)

table2:
	$(PY) scripts/run_utility.py    --config $(MODEL) $(EXTRA)

table3:
	$(PY) scripts/run_efficiency.py --config $(MODEL) $(EXTRA)

table4:
	$(PY) scripts/run_ablation.py   --config $(MODEL) $(EXTRA)

sweeps:
	$(PY) scripts/run_sweep.py      --config $(MODEL) $(EXTRA)

tables:
	$(PY) scripts/make_tables.py

figures:
	$(PY) scripts/make_figures.py

# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
smoke:
	$(PY) scripts/smoke_test.py --config $(MODEL) --override $(SMOKE)

all: table1 table2 table3 table4 sweeps tables figures

all-models:
	@for m in $(MODELS); do \
	  echo "=== $$m ==="; \
	  $(MAKE) all MODEL=$$m || exit 1; \
	done

clean:
	rm -rf results/*/table*.json results/*/sweeps.json results/*/fig*.p*
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
