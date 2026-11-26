FROM python:3.11-slim

# Create a non-root user
RUN useradd -m -s /bin/bash appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files safely
COPY src/ ./src/
# By copying app.py alongside the optional JSONs, the COPY command will never fail
COPY app.py baseline_metrics.json* lda_topics.json* transformer_metrics.json* accuracy_chart.png* class_metrics_chart.png* ./

# Change ownership and switch to non-root user
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 8501

# Native Python Healthcheck with timeout
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
