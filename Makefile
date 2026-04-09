.PHONY: help setup up down restart logs test lint format parquet

help:
	@echo "Available commands:"
	@echo "  make setup      - Copy .env.example to .env"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - Show logs (use svc=<service> to filter)"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run ruff linter"
	@echo "  make format     - Format code with ruff"
	@echo "  make parquet    - Convert raw CSV.gz to Parquet (run once)"
	@echo "  make train      - Trigger training DAG manually"
	@echo "  make predict    - Test API prediction endpoint"

setup:
	cp .env.example .env
	mkdir -p credentials models data/raw data/processed data/images logs

up:
	docker compose up -d --build
	@echo "Airflow: http://localhost:8080 (admin/admin)"
	@echo "MLflow:  http://localhost:5000"
	@echo "API:     http://localhost:8000/docs"

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f $(svc)

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ dags/ tests/

format:
	ruff format src/ dags/ tests/

parquet:
	python scripts/convert_to_parquet.py

train:
	docker compose exec airflow-scheduler airflow dags trigger training_dag

predict:
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"neighbourhood": "Pinheiros", "room_type": "Entire home/apt", "accommodates": 4, "bathrooms": 1.0, "bedrooms": 2, "beds": 2, "minimum_nights": 2, "number_of_reviews": 50, "review_scores_rating": 4.8, "amenities": ["wifi", "kitchen", "air conditioning"]}'
