FROM python:3.12-slim

# Create a non-root user
RUN useradd -m -s /bin/bash appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit

# Copy files safely
COPY src/ ./src/
COPY app.py .
COPY baseline_metrics.json* lda_topics.json* transformer_metrics.json* ./

# Change ownership and switch to non-root user
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 8501

# Native Python Healthcheck
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health') or exit(1)"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
