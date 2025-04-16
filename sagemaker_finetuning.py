# -*- coding: utf-8 -*-
"""
following are steps:

1. Setup dependencies and paths
2. Build toy dataset in the prompt-response format
3. Create SageMaker Training job

## Setup dependencies and paths
"""

!pip install "sagemaker==2.162.0" s3path --quiet

from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from sagemaker import s3_utils
import sagemaker
import boto3
import json

# Define S3 paths
bucket             = "<YOUR-S3-BUCKET>"
training_data_path = f"s3://{bucket}/toy_data/train/data.jsonl"
test_data_path     = f"s3://{bucket}/toy_data/test/data.jsonl"
output_path        = f"s3://{bucket}/outputs"
code_location      = f"s3://{bucket}/code"

# Create SageMaker session
sagemaker_session  = sagemaker.Session()
region             = sagemaker_session.boto_region_name
role               = sagemaker.get_execution_role()

"""## 2. Build toy dataset in the prompt-response format

We will be fine-tuning the model on this small, toy dataset for demonstration purposes.
"""

prompt_template = """Write a response that appropriately answers the question below.
### Question:
{question}

### Response:
"""

dataset = [
    {"prompt": "What is a Pastel de Nata?",
     "response": "A Pastel de Nata is a Portuguese egg custard tart pastry, optionally dusted with cinnamon."},
    {"prompt": "Which museums are famous in Amsterdam?",
     "response": "Amsterdam is home to various world-famous museums, and no trip to the city is complete without stopping by the Rijksmuseum, Van Gogh Museum, or Stedelijk Museum."},
    {"prompt": "Where is the European Parliament?",
     "response": "Strasbourg is the official seat of the European Parliament."},
    {"prompt": "How is the weather in The Netherlands?",
     "response": "The Netherlands is a country that boasts a typical maritime climate with mild summers and cold winters."},
    {"prompt": "What are Poffertjes?",
     "response": "Poffertjes are a traditional Dutch batter treat. Resembling small, fluffy pancakes, they are made with yeast and buckwheat flour."},
]

# Format prompt based on template
for example in dataset:
    example["prompt"] = prompt_template.format(question=example["prompt"])

training_data, test_data = dataset[0:4], dataset[4:]

print(f"Size of training data: {len(training_data)}\nSize of test data: {len(test_data)}")

"""#### Upload training and test data to S3"""

def write_jsonlines_to_s3(data, s3_path):
    """Writes list of dictionaries as a JSON lines file to S3"""

    json_string = ""
    for d in data:
        json_string += json.dumps(d, ensure_ascii=False) + "\n"

    s3_client   = boto3.client("s3")

    bucket, key = s3_utils.parse_s3_url(s3_path)
    s3_client.put_object(
         Body   = json_string,
         Bucket = bucket,
         Key    = key,
    )

write_jsonlines_to_s3(training_data, training_data_path)
write_jsonlines_to_s3(test_data, test_data_path)

"""## 3. Create SageMaker training job

### 3.1. Prepare training entry point script

We will be using the general-purpose training script from llm-foundry and using Composer to set up the environment for distributed training.
"""


"""### 3.2. Define 🤗 HuggingFace estimator"""

# Define container image for the training job
training_image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04-v1.1"

# Define metrics to send to CloudWatch
metrics = [
    # On training set
    {"Name": "train:LanguageCrossEntropy",
     "Regex": "Train metrics\/train\/LanguageCrossEntropy: ([+-]?((\d+\.?\d*)|(\.\d+)))"},
    {"Name": "train:LanguagePerplexity",
     "Regex": "Train metrics\/train\/LanguagePerplexity: ([+-]?((\d+\.?\d*)|(\.\d+)))"},
    # On test set
    {"Name": "test:LanguageCrossEntropy",
     "Regex": "Eval metrics\/eval\/LanguageCrossEntropy: ([+-]?((\d+\.?\d*)|(\.\d+)))"},
    {"Name": "test:LanguagePerplexity",
     "Regex": "Eval metrics\/eval\/LanguagePerplexity: ([+-]?((\d+\.?\d*)|(\.\d+)))"},
]

estimator_args = {
    "image_uri": training_image_uri,     # Training container image
    "entry_point": "launcher.sh",        # Launcher bash script
    "source_dir": ".",                   # Directory with launcher script and configuration file
    "instance_type": "ml.g5.48xlarge",   # Instance type
    "instance_count": 1,                 # Number of training instances
    "base_job_name": "fine-tune-mpt-7b", # Prefix of the training job name
    "role": role,                        # IAM role
    "volume_size": 300,                  # Size of the EBS volume attached to the instance (GB)
    "py_version": "py310",               # Python version
    "metric_definitions": metrics,       # Metrics to track
    "output_path": output_path,          # S3 location where the model artifact will be uploaded
    "code_location": code_location,      # S3 location where the source code will be saved
    "disable_profiler": True,            # Do not create profiler instance
    "keep_alive_period_in_seconds": 240, # Enable Warm Pools while experimenting
}

huggingface_estimator = HuggingFace(**estimator_args)

"""### 3.3. Start training job with the training and test datasets uploaded to S3"""

huggingface_estimator.fit({
    "train": TrainingInput(
        s3_data=training_data_path,
        content_type="application/jsonlines"),
    "test": TrainingInput(
        s3_data=test_data_path,
        content_type="application/jsonlines"),
}, wait=True)

"""## Have fun with your fine-tuned MPT-7B model!"""
