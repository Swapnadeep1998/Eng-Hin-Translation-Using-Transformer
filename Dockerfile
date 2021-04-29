FROM continuumio/miniconda3
COPY . /app
WORKDIR /app
RUN conda env create -f environment.yml --force
SHELL ["conda", "run", "-n", "bertQA", "/bin/bash", "-c"]
RUN python -c "import tensorflow as tf"
EXPOSE 8000
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bertQA", "python", "app.py"]