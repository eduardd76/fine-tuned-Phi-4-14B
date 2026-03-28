# Phi-4 Network Architect — Make targets
# Usage: make <target>

.PHONY: help setup data train eval quantize deploy test clean

help:
	@echo "Phi-4 Network Architect Pipeline"
	@echo ""
	@echo "  make setup       Install dependencies"
	@echo "  make data        Generate 10k training samples"
	@echo "  make train       Fine-tune with Unsloth LoRA"
	@echo "  make eval        Run evaluation suite"
	@echo "  make quantize    Quantize (GPTQ + GGUF)"
	@echo "  make deploy      Start API server"
	@echo "  make test        Run all tests"
	@echo "  make aws-launch  Launch EC2 instance"
	@echo "  make docker-up   Start Docker stack"
	@echo "  make clean       Remove generated files"

VENV = $(HOME)/phi4-env
PYTHON = $(VENV)/bin/python
DATA_DIR = /data
MODEL_DIR = $(DATA_DIR)/models/phi4-network-architect

setup:
	python3.11 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121
	$(VENV)/bin/pip install -q -r fine_tuning/requirements.txt
	@echo "Setup complete. Activate: source $(VENV)/bin/activate"

data:
	$(PYTHON) data_generation/dataset_generator.py \
		--count 10000 \
		--output $(DATA_DIR)/generated_data
	$(PYTHON) data_generation/split.py \
		--input $(DATA_DIR)/generated_data/training_data.jsonl \
		--stratify

train:
	@mkdir -p $(DATA_DIR)/checkpoints $(DATA_DIR)/logs
	$(PYTHON) fine_tuning/train.py \
		--train-data $(DATA_DIR)/generated_data/training_data.jsonl \
		--val-data   $(DATA_DIR)/generated_data/validation_data.jsonl \
		--output-dir $(DATA_DIR)/checkpoints \
		2>&1 | tee $(DATA_DIR)/logs/training.log

train-resume:
	$(PYTHON) fine_tuning/train.py \
		--resume $(DATA_DIR)/checkpoints \
		--train-data $(DATA_DIR)/generated_data/training_data.jsonl \
		--val-data   $(DATA_DIR)/generated_data/validation_data.jsonl \
		--output-dir $(DATA_DIR)/checkpoints

eval:
	$(PYTHON) evaluation/run_all.py \
		--model $(DATA_DIR)/checkpoints \
		--test-data evaluation/test_cases.jsonl \
		--output $(DATA_DIR)/logs/eval_results.json

quantize:
	$(PYTHON) deployment/quantize_gptq.py \
		--model $(DATA_DIR)/checkpoints \
		--output $(DATA_DIR)/models/phi4-gptq
	$(PYTHON) deployment/quantize_gguf.py \
		--model $(DATA_DIR)/checkpoints \
		--output $(DATA_DIR)/models/phi4-gguf \
		--quant-type Q4_K_M
	ln -sfn $(DATA_DIR)/models/phi4-gptq $(MODEL_DIR)

deploy:
	$(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

deploy-mcp:
	$(PYTHON) dream_team_integration/mcp_server.py

test:
	$(VENV)/bin/pytest tests/ -v --tb=short 2>&1 | tee $(DATA_DIR)/logs/test_results.log

test-unit:
	$(VENV)/bin/pytest tests/unit/ -v

test-integration:
	$(VENV)/bin/pytest tests/integration/ -v -s

aws-launch:
	bash aws_deployment/launch_ec2.sh

aws-monitor:
	@read -p "EC2 IP: " ip; bash aws_deployment/monitoring.sh $$ip

docker-up:
	cd deployment_artifacts/docker && docker-compose up -d

docker-down:
	cd deployment_artifacts/docker && docker-compose down

pipeline:
	bash run_full_pipeline.sh

clean:
	rm -rf $(DATA_DIR)/generated_data/*.jsonl
	rm -rf __pycache__ **/__pycache__
	@echo "Cleaned generated data. Models preserved."
