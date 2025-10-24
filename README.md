[![GenBio AI](https://raw.githubusercontent.com/genbio-ai/ModelGenerator/main/docs/docs/assets/images/genbio_header.png)](https://genbio.ai/)

# ModelGenerator Tutorial: AIDO Foundation Models

This repository hosts a hands-on tutorial for GenBio AI's ModelGenerator library. The notebooks walk through environment setup, notebook-based exploration, and command line fine-tuning workflows for three flagship foundation models released under the GenBio AI Community License.

## Highlights

- Reproducible setup instructions for Python 3.12 with CUDA-enabled PyTorch
- Guided notebooks covering DNA, protein, and protein-RAG model families
- Step-by-step breakdown of ModelGenerator configuration files and Lightning integration
- Ready-to-run CLI commands for LoRA and distributed training recipes
- Links to documentation, datasets, and model cards for deeper exploration

## Prerequisites

- Conda or mamba for environment management
- Python 3.12
- CUDA 11.8 capable machine for GPU workflows (optional but recommended)
- Git, pip, and access to Hugging Face Hub

## Quick Start

```bash
# Create and activate the tutorial environment
conda create -n genbio python=3.12 -y
conda activate genbio

# Install core dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip install datasets==3.0.0 hf_transfer tabulate

# Clone and install ModelGenerator in editable mode
git clone https://github.com/genbio-ai/ModelGenerator.git
cd ModelGenerator
pip install -e .
```

## Repository Layout

- `Notebooks/` interactive tutorials and sample assets
- `ModelGenerator/` embedded copy of the upstream library for reference
- `ModelGenerator/experiments/` production-ready training configs
- `ModelGenerator/docs/` MkDocs documentation sources
- `ModelGenerator/tests/` automated regression suite

## Tutorial Notebooks

- `Notebooks/Install-Env.ipynb` end-to-end environment bootstrap and sanity checks
- `Notebooks/Protein-16B.ipynb` loading, tokenizing, and embedding sequences with AIDO.Protein-16B
- `Notebooks/Protein-16B-task.ipynb` configuring LoRA adapters and Lightning training loops on ProteinGym DMS tasks
- `Notebooks/Protein-RAG-16B.ipynb` multimodal inference combining MSA and structure tokens with AIDO.Protein-RAG-16B
- `Notebooks/Protein-RAG-16B-task.ipynb` distributed fine-tuning recipes for retrieval-augmented protein models
- `Notebooks/DNA-300M.ipynb` (and task variant) genome-scale language modeling walkthrough for AIDO.DNA-300M

## Featured Models

### AIDO.DNA-300M

- 300 million parameter DNA foundation model trained on 10.6 billion nucleotides across 796 species
- Supports genome mining, in silico mutagenesis, expression prediction, and directed sequence design
- Notebook highlights tokenization, embedding inspection, and downstream regression tasks

### AIDO.Protein-16B

- 16 billion parameter mixture-of-experts protein language model trained on 1.2 trillion amino acids
- Demonstrates sequence embeddings, classification, and regression heads with ModelGenerator adapters
- Includes LoRA fine-tuning workflow on ProteinGym Deep Mutational Scanning benchmark

### AIDO.Protein-RAG-16B

- Multimodal extension of AIDO.Protein-16B that fuses MSA context and structure embeddings
- Tutorials cover structure tokenization, 2D positional encoding, and Lightning training with custom callbacks
- Provides end-to-end example spanning preprocessing, inference, and evaluation

## Training with the ModelGenerator CLI

ModelGenerator exposes an `mgen` command that orchestrates data modules, backbones, adapters, and Lightning trainers from YAML configs. Example LoRA fine-tuning run:

```bash
export HF_HOME=/tmp/hf_cache

TASK_NAME="A4GRB6_PSEAI_Chen_2020"
MUTATION_TYPE="singles_substitutions"
RUN_NAME="${TASK_NAME}_fold0"

mgen fit \
  --config experiments/AIDO.Protein/DMS/configs/substitution_LoRA_DDP.yaml \
  --data.train_split_files "[\"${MUTATION_TYPE}/${TASK_NAME}.tsv\"]" \
  --data.cv_test_fold_id 0 \
  --data.batch_size 2 \
  --trainer.logger.project AIDO_Demo \
  --trainer.logger.name ${RUN_NAME} \
  --trainer.devices auto
```

Refer to the task notebooks for a line-by-line reconstruction of the underlying Python objects used by each configuration.

## Additional Resources

- Documentation: [https://genbio-ai.github.io/ModelGenerator/](https://genbio-ai.github.io/ModelGenerator/)
- Hugging Face collection: [https://huggingface.co/genbio-ai](https://huggingface.co/genbio-ai)
- GitHub organization: [https://github.com/genbio-ai](https://github.com/genbio-ai)
- ProteinGym benchmark: [https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1)
- AIDO.Protein MoE paper: [https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1](https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1)
- AIDO.Protein-RAG paper: [https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1](https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1)

## License

Unless otherwise noted, code and assets in this tutorial are distributed under the GenBio AI Community License Agreement. See `ModelGenerator/LICENSE` for the full terms.
