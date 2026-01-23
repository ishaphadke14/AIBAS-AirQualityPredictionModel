FROM busybox
RUN mkdir -p /tmp/knowledgeBase
WORKDIR /knowledgeBase
COPY ../code/Models/currentAiSolution.keras /knowledgeBase/
COPY ../code/Models/currentOLSsolution.pkl /knowledgeBase/
COPY images/knowledgeBase_AirQualityIndexPredictionModel/Readme.md /knowledgeBase/Readme.md

