det command run --config-file config-retail.yaml --context . "pip install streamlit-datalist weaviate-client && mkdir -p .streamlit && cp secrets.toml .streamlit/secrets.toml && streamlit run qa_retail.py --server.port 443"