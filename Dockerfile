FROM gcr.io/broad-getzlab-workflows/base_image:v0.0.6 
# python 3.8

WORKDIR /build

RUN pip install --upgrade pip
RUN pip install AnnoMate==1.0.1

# Install repository to get tutorial notebooks and example data
RUN git clone https://github.com/getzlab/AnnoMate.git

# install simulated data for example notebooks for prebuilt reviewers
RUN git clone https://github.com/getzlab/SimulatedTumorData
RUN pip install -e SimulatedTumorData/.
