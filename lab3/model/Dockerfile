FROM pytorch/pytorch:latest

USER root

WORKDIR /app/model
COPY . /app/model

# Install the requirements
RUN pip3 install --upgrade pip && \
    pip3 install -Ur requirements.txt

# Run the pipeline
RUN chmod +x /app/model/pipeline.sh
CMD ["/app/model/pipeline.sh"]
