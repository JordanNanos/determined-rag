docker run -it --gpus '"device=1"' \
	-e TAKEOFF_MODEL_NAME=NousResearch/Llama-2-7b-chat-hf \
	-e TAKEOFF_DEVICE=cuda \
	-e TAKEOFF_BACKEND=fast \
	-p 80:3000 \
	-p 8080:3001 \
    -v /root/.iris_cache:/code/models \
	tytn/takeoff-pro:0.6.2-gpu

#	-p 3001:3001 \
#	-p 3000:3000 \
#-e TAKEOFF_MODEL_NAME=TitanML/llama2-7b-chat-4bit-AWQ \
#-e TAKEOFF_MODEL_NAME=TitanML/ct2-bfloat16-Llama-2-7b-chat-hf \
