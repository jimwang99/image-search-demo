export PYTHONPATH := $(shell pwd)/pysrc

.PHONY: download-image
download-images: data/val2017.zip
data/inputs/val2017.zip:
	mkdir -p data/inputs \
		&& cd data/inputs \
		&& wget http://images.cocodataset.org/zips/val2017.zip \
		&& unzip val2017.zip

.PHONY: lint
lint:
	ruff --config pyproject.toml check .
	
.PHONY: test
test:
	python -m pytest tests