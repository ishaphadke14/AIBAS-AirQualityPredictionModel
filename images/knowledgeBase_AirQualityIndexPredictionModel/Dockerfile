FROM busybox
RUN mkdir -p /tmp/knowledgeBase
WORKDIR /knowledgeBase
COPY ../code/Models/currentAiSolution.h5 /knowledgeBase/
COPY ../code/Models/currentOLSsolution.pkl /knowledgeBase/
COPY images/knowledgeBase_AirQualityIndexPredictionModel/Readme.md /knowledgeBase/Readme.md

COPY ../code/Models/scaler_X.pkl /knowledgeBase/
COPY ../code/Models/scaler_y.pkl /knowledgeBase/