# A Mixture of Post-training Datasets to Enhance the Scientific Reasoning Capability of Large Language Models

## Data Process

### Step 0. Install Environment

```bash
cd data_process
conda create --name data_mixture python=3.10
conda activate data_mixture
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -U pynvml
```

Launch the Python interpreter and download the necessary NLTK tokenizer data:

```python
python -c "import nltk; nltk.download('punkt_tab')"
```

Alternatively, you can run this interactively:
```python
python
>>> import nltk
>>> nltk.download('punkt_tab')
>>> exit()
```

### Step 1. Question Deduplication

We employ [text-dedup](https://github.com/ChenghaoMou/text-dedup) to remove duplicate questions from datasets. This tool provides efficient text deduplication capabilities using various algorithms including MinHash, SimHash, and exact hash matching.

#### Environment Setup

First, create and activate a dedicated conda environment:
```bash
conda create --name textdedup python=3.10
conda activate textdedup
pip install text-dedup
```

#### Configuration

Configure the deduplication parameters in the script:
```bash
script/question_dedup.sh
```

#### Execution

Run the deduplication script:
```bash
bash script/question_dedup.sh
```

#### Output

After processing, merge all deduplicated chunks into a single JSONL file for downstream use.

### Step 2. LLM-based Question Decontamination

To ensure data quality and prevent benchmark contamination, we implement a comprehensive decontamination pipeline using embedding-based similarity search followed by LLM-based semantic judgment.

#### Generate Benchmark Embeddings

First, generate embeddings for benchmark questions to create a searchable index:

```bash
python decontamination/benchmark_index_save.py \
    --model BAAI/bge-large-en-v1.5 \
    --batch_size 1024 \
    --output_path decontamination/index/benchmark_embedding.jsonl
```

#### Generate Dataset Embeddings

Generate embeddings for your dataset questions:

```bash
python decontamination/data_index_save.py \
    --model BAAI/bge-large-en-v1.5 \
    --batch_size 1024 \
    --input_path dataset_path.jsonl \
    --output_path decontamination/index/data_embedding.jsonl
```

#### Similarity Search

Perform vector similarity search to identify potentially similar questions between your dataset and benchmark:

```bash
python decontamination/vector_search.py \
    --data_embedding_path decontamination/index/data_embedding.jsonl \
    --benchmark_embedding_path decontamination/index/data_embedding.jsonl \
    --output_path decontamination/results/dataset_with_top5_similarity_benchmark.jsonl
```

#### LLM-based Similarity Judge

Configure the decontamination process by modifying the task configuration:g `data_process/vllm_inference/task_config/llm_based_decontamination.yaml`.

Execute the LLM-based similarity judgment:

```bash
bash script/llm_based_decontamination.sh
```

After the LLM judgment completes, run the post-processing step to generate the final decontaminated dataset:

```bash
python vllm_inference/llm_based_decontamination_postprocess.py \
    --input_data_dir data/llm_based_decontamination/original_data \
    --output_path data/llm_based_decontamination/final_data/dataset_decontamination.jsonl
```

### Step 3. Random Selection

Change the input path and output path in `data_selection_by_random.py`, then run:

```bash
python data_selection_by_random.py
```

### Step 4. Response Length Selection

Change the input path and output path in `data_selection_by_answer_length.py`, then run:

```bash
python data_selection_by_answer_length.py
```

### Step 5. Difficulty Selection

Extract reference answer:

```bash
bash script/extract_reference_answer.sh
```

Distill answer:

```bash
bash script/distill_answer.sh
```

Data Selection:

```bash
python data_selection_by_eval_score_mp.py
```

## Training

`cd LLaMA-Factory`

Following [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to set up the environment.

Change path and superparameters in `yaml` and `bash_scripts`.

run bash scripts:

```bash
bash bash_scripts/data_mixture.sh
```

## Evaluation

`cd lm-open-science-evaluation`

Following [Language Model Open Science Evaluation](https://github.com/GAIR-NLP/lm-open-science-evaluation) to set up the environment.

```bash
bash scripts/eval_science.sh <model_path>
```