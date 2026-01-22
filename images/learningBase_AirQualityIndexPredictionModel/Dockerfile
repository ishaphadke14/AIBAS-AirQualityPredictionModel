FROM busybox

RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY ../data/processed/training_data.csv /tmp/learningBase/train/
COPY ../data/processed/test_data.csv /tmp/learningBase/validation/

COPY images/learningBase_AirQualityIndexPredictionModel/Readme.md /tmp/learningBase/Readme.md



