.Phony: all

PROG=python3

MAIN=chatbot
MAINPY=$(MAIN).py
SCRIPTS=tests
MODEL=chatbot
INTENTS=intents
ASSETS=assets
JSON=$(ASSETS)/$(INTENTS).json

all: train

requirements:
	$(PROG) -m pip install -r requirements.txt

train: $(JSON)
	@$(PROG) $(MAINPY) -n $(MODEL) $(JSON)

tests: test_patterns test_brain clean

test_patterns:
	@$(SCRIPTS)/test_prog test_patterns

test_brain:
	@$(SCRIPTS)/test_prog test_brain

clean:
	@rm -f *.pkl *.h5
	@py3clean .
