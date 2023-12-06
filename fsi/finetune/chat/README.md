# Usage

Training can be run on Grenoble using `det e create configs/mpt-7b-grenoble.yaml .`

To run training on a new cluster, you must first prepare data on a shared filesystem using the `prepare_data.py` script.  Then, change the `--data_files` argument to point to the new data location.  Cluster specific configuration options such as `resource_pool` and cache directories should also be changed.

If re-initializing the huggingface cache, a patch is needed in the MPT-7B `blocks.py` file when using torch < 1.12 -- delete the `approximate` argument to `GELU`.