#det command run --context . --config-file config.yaml "python prepare_chunks.py --tokenizer 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5' --hf-cache-dir '/cstor/coreystaten/hf' --source-dir '/cstor/anoop/db/data-scraping' --output-dir '/cstor/coreystaten/deutsche/data/scraped-csv/'"
#det command run --context . --config-file config.yaml "python prepare_chroma_index.py --tokenizer '/cstor/liam/models/SGPT-5.8B-finetuned-v2' --embedding-model '/cstor/liam/models/SGPT-5.8B-finetuned-v2' --hf-cache-dir '/cstor/coreystaten/hf' --chunks-file /cstor/coreystaten/deutsche/data/scraped-csv/train_chunks.csv --chunks-file /cstor/coreystaten/deutsche/data/scraped-csv/test_chunks.csv --output /cstor/coreystaten/deutsche/data/indices/sgpt-5.8b-finetuned-retrieval-v2 --prompt '{}' --tokenize-logic sgpt"
#det command run --context . --config-file config.yaml "python prepare_weaviate_index.py --tokenizer '/cstor/liam/models/SGPT-5.8B-finetuned-v2' --embedding-model '/cstor/liam/models/SGPT-5.8B-finetuned-v2' --hf-cache-dir '/cstor/coreystaten/hf' --chunks-file /cstor/coreystaten/deutsche/data/scraped-csv/train_chunks.csv --chunks-file /cstor/coreystaten/deutsche/data/scraped-csv/test_chunks.csv --output /cstor/coreystaten/deutsche/data/weaviate/sgpt-5.8b-finetuned-retrieval-v5 --prompt '{}' --tokenize-logic sgpt"
#det command run --context . --config-file config.yaml "python prepare_qpa.py --result-source-dir /cstor/coreystaten/deutsche/data/qpa/raw --chunk-source-dir /cstor/coreystaten/deutsche/data/scraped-csv/ --output-dir /cstor/coreystaten/deutsche/data/qpa/parsed-v2/"


#det command run --context . --config-file config.yaml "python prepare_weaviate_index.py --tokenizer '/root/.hfcache/SGPT-5.8B-finetuned-v2' --embedding-model '/root/.hfcache/SGPT-5.8B-finetuned-v2' --hf-cache-dir '/root/.hfcache' --chunks-file /root/dt-rag/fsi/data/chunks/train_chunks.csv --chunks-file /root/dt-rag/fsi/data/chunks/test_chunks.csv --output /root/dt-rag/fsi/data/output/sgpt-5.8b-finetuned-retrieval-v5 --prompt '{}' --tokenize-logic sgpt"



det command run --context . --config-file config.yaml "python prepare_chunks.py --tokenizer 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5' --hf-cache-dir '/home/l40s/.hfcache' --source-dir '/home/l40s/data-scraping' --output-dir '/home/l40s/dt-rag/telco/data/chunks'

det command run --context . --config-file config.yaml "python prepare_weaviate_index.py --tokenizer '/home/l40s/.hfcache/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit' --embedding-model '/root/.hfcache/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit' --hf-cache-dir '/home/l40s/.hfcache' --chunks-file /root/dt-rag/telco/data/chunks/train_chunks.csv --chunks-file /home/l40s/dt-rag/telco/data/chunks/test_chunks.csv --output /home/l40s/dt-rag/telco/data/output/ --prompt '{}' --tokenize-logic sgpt"



