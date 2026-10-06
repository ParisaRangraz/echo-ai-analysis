# Inference image: runs infer.py with an ONNX model. No PyTorch, so it stays small.
#
# Build:   docker build -t echo-ai-analysis .
# Run:     docker run --rm \
#              -v "$PWD/models:/models:ro" -v "/path/to/patient:/input:ro" \
#              echo-ai-analysis --model /models/pvt_rta.onnx \
#              --seq-4ch /input/patient0001_4CH_half_sequence.nii.gz \
#              --seq-2ch /input/patient0001_2CH_half_sequence.nii.gz \
#              --cfg-4ch /input/Info_4CH.cfg --cfg-2ch /input/Info_2CH.cfg
#
# The model and the patient data are mounted, not baked in: checkpoints are not
# in the repository, and the CAMUS terms do not allow redistributing the data.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-infer.txt .
RUN pip install -r requirements-infer.txt

# Only the files infer.py imports.
COPY infer.py .
COPY data/loader.py data/processing.py data/
COPY evaluation/volume.py evaluation/

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["python", "infer.py"]
CMD ["--help"]
