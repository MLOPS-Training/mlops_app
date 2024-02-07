# mlops_app

## Description

This is a simple MLOps application that demonstrates the use of CI/CD pipelines to deploy a machine learning model. The application is built using the following technologies:

- Python
- Flask
- Docker

## Development

To run the application in development mode, you can run the following commands:

```bash
cd mlops_app
python src/app.py
```

## Installation

To install the application, you need to have the following installed:

- Python
- Docker

To install the application, you can clone the repository and run the following commands:

```bash
cd mlops_app
docker build -t mlops_app .
docker run -p 5000:5000 mlops_app
```

## Usage

To use the application, you can visit the following URL:

```bash
http://localhost:5000
```

## Contributors

- [Cesar Ombredane](https://github.com/cesarombredane)
- [Meryem Koze](https://github.com/mrykse)
- [Arthur Nguyen](https://github.com/NguyenArthur)
