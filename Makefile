.PHONY: test train train-all app clean

PYTHON := PYTHONPATH=. python

test:
	$(PYTHON) -m pytest tests/ -q

train:
	$(PYTHON) -m src.train --subset FD001 --model all

train-all:
	$(PYTHON) -m src.train --subset FD001 --model all
	$(PYTHON) -m src.train --subset FD002 --model xgb,lstm
	$(PYTHON) -m src.train --subset FD003 --model xgb,lstm
	$(PYTHON) -m src.train --subset FD004 --model xgb,lstm

app:
	PYTHONPATH=. streamlit run app/streamlit_app.py

clean:
	rm -rf models/*.joblib models/*.pt models/*.json data/processed/* reports/results.csv
