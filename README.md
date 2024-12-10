# Image Search Demo

This is a demo for image search use LLM to improve its accuracy.

## How to use

### Setup Python

1. Install [miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)
2. Create virtual env for this demo and install required packages
```
conda create -n image-search-demo python=3.12
conda activate image-search-demo
pip install -r requirements.txt
```

### Prepare the repo

```
git clone https://github.com/jimwang99/image-search-demo.git
cd image-search-demo
```

### Prepare dataset

```
make download-images
```
This step downloads 5K coco-2017 validation dataset from https://cocodataset.org, and it will take some time.

### Initialize database

```
make init
```
This step generates embeddings for all the images from above dataset, and inserts them into the database. It will take some time (about 30mins on my MacBook Air with Apple M2 CPU/GPU).

### Run

```
make run
```

Use browser to open [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

## Features

- Support add/remove image to/from the database
- Support image search with natural language queries
- Support image search by uploading image

### Non-Goal

- User/admin authentication


## Documents

### [Architecture](./docs/architecture.md)
### [Implementation](./docs/implementation.md)
### [Design Choices](./docs/design-choices.md)
### [Future Work](./docs/future-work.md)

## TODOs

- [x] Skeleton project
- [x] High-level diagrams
- [x] AI stack choices
- [x] Software stack choices
- [x] Detailed diagram
- [x] 1st review
- [x] Skeleton code with unit-tests
- [x] Complete documents
- [x] More integration tests
- [x] Performance optimization
- [x] MVP demo
- [ ] 2nd review
- [ ] Improvements
