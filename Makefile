.PHONY: download-image
download-images: data/val2017.zip
data/images/val2017.zip:
	mkdir -p data/images \
		&& cd data/images \
		&& wget http://images.cocodataset.org/zips/val2017.zip \
		&& unzip val2017.zip
	