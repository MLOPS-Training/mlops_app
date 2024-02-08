FROM python:3.8-slim
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements_unix.txt
EXPOSE 5000
CMD ["python3", "-u", "./src/app.py", "--prod"]
