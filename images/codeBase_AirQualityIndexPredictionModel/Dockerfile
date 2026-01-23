FROM busybox
FROM python:3.10-slim
RUN mkdir -p /tmp/codeBase
WORKDIR /codeBase
COPY /images/codeBase_AirQualityIndexPredictionModel/RunAIModel.py /codeBase/RunAIModel.py
COPY /images/codeBase_AirQualityIndexPredictionModel/RunOLSModel.py /codeBase/RunOLSModel.py
RUN pip install --no-cache-dir numpy pandas tensorflow==2.18.0 keras statsmodels
COPY images/knowledgeBase_AirQualityIndexPredictionModel/Readme.md /knowledgeBase/Readme.md
# CMD ["python", "/tmp/codeBase/RunAIModel.py"]



