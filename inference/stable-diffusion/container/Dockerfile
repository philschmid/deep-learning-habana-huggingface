FROM vault.habana.ai/gaudi-docker/1.7.0/ubuntu20.04/habanalabs/pytorch-installer-1.12.0:latest

# Install dependencies
COPY app/requirements.txt  ./requirements.txt
RUN python3 -m pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY app/main.py  ./main.py
CMD ["python3", "-m", "main:app", "--workers", "8"]
