FROM gcr.io/broad-getzlab-workflows/base_image:v0.0.6 
# python 3.8

WORKDIR /build

RUN pip install --upgrade pip
RUN pip install AnnoMate==1.0.0

# Install prebuilt reviewer environments
RUN git clone https://github.com/getzlab/AnnoMate.git