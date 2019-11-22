FROM alegeno92/opencv_python3:3.4.2
RUN pip install paho-mqtt
RUN pip install flask
RUN mkdir /app
COPY . /app/
ENV PYTHONUNBUFFERED=1
CMD ["python3", "/app/streamer.py", "/config/config.json"]
