pip install -r requirements.txt
git clone -b determined https://github.com/liamcli/sgpt
cd sgpt/sentence-transformers
pip install einops
pip install -e .
cd sentence_transformers/losses/GradCache; pip install -e .

cd /run/determined/workdir
