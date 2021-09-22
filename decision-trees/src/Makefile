VENV =venvCS60050
.PHONY: clean run 
PYTHON =$(VENV)/Scripts/python
PIP =$(VENV)/Scripts/pip

run:$(VENV)/Scripts/activate
	$(PYTHON) "main.py"


$(VENV)/Scripts/activate: requirements.txt
	virtualenv $(VENV)
	$(PIP) install --no-cache-dir -r requirements.txt


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -rf *.dot
