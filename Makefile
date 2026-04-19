PYTHON ?= python

.PHONY: serve eval perf prepare improve guardrails all

serve:
	$(PYTHON) serve/serve.py

eval:
	$(PYTHON) eval_runner/run_eval.py --limit 30

perf:
	$(PYTHON) perf/load_test.py

prepare:
	$(PYTHON) improve/prepare_data.py

improve:
	$(PYTHON) improve/infer.py

guardrails:
	$(PYTHON) guardrails/validate.py

all: eval perf prepare improve guardrails