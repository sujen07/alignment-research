# GPU recommended. Run with: docker run --gpus all ... and mount the docker socket
# so the runner can `docker exec` into fake-target, same as the original experiment.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
# default: train then evaluate
CMD ["sh", "-c", "python train_grpo.py && python eval_rl.py"]
