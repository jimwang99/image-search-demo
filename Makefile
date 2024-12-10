export PYTHONPATH := $(shell pwd)/pysrc

.PHONY: download-images
download-images: data/inputs/val2017.zip
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
	pytest --ignore=data --doctest-modules -s
	rm -rf data/test

.PHONY: doc
doc:
	pdoc -d google --no-include-undocumented -o docs/api pysrc/embedding_server.py pysrc/database_server.py pysrc/image_server.py pysrc/backend.py pysrc/config.py

.PHONY: init
init: download-images
	python pysrc/backend.py

.PHONY: run
run:
	python pysrc/frontend.py