FROM determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.22.1
#ENV http_proxy=http://web-proxy.bbn.hpecorp.net:8088
#ENV https_proxy=http://web-proxy.bbn.hpecorp.net:8088
#WORKDIR /app
#RUN mkdir .streamlit
#COPY secrets.toml .streamlit/secrets.toml
COPY requirements.txt .
RUN pip install -r requirements.txt
#COPY qa.py .