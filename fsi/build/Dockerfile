FROM anoophpe/app-qa:4.0
RUN pip install weaviate-client==3.19.2
WORKDIR /app
COPY .streamlit/secrets.toml .streamlit/secrets.toml
COPY .streamlit/config.toml .streamlit/config.toml
COPY style.css .
COPY qa_weaviate.py .