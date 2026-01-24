FROM busybox
FROM python:3.10-slim
RUN mkdir -p /tmp/codeBase
WORKDIR /codeBase
COPY /images/codeBase_AirQualityIndexPredictionModel/RunAIModel.py /codeBase/RunAIModel.py
COPY /images/codeBase_AirQualityIndexPredictionModel/RunOLSModel.py /codeBase/RunOLSModel.py
RUN pip install --no-cache-dir --default-timeout=1000 --upgrade --upgrade-strategy only-if-needed numpy pandas tensorflow==2.15.1 "keras<3" statsmodels joblib scikit-learn
COPY images/codeBase_AirQualityIndexPredictionModel/Readme.md /codeBase/Readme.md
# CMD ["python", "/tmp/codeBase/RunAIModel.py"]
