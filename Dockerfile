FROM agrigorev/zoomcamp-bees-wasps:v2


#RUN pip --no-cache-dir install pipenv

COPY lambda_function.py .
#RUN set -ex && pipenv install --deploy --system
#RUN pipenv install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install Pillow 
#RUN pip install urllib
#RUN pip install io
#CMD [ "pipenv run lambda_function.lambda_handler" ]


ENV MODEL_NAME=bees-wasps-v2.tflite

CMD [ "pip run lambda_function.lambda_handler" ]