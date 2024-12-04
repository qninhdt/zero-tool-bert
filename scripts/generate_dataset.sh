mkdir -p datasets
mkdir -p datasets/apigen
mkdir -p datasets/glaive
mkdir -p datasets/toolace

# Download datasets
# wget -O datasets/apigen/xlam_function_calling_60k.json https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/resolve/main/xlam_function_calling_60k.json
echo "APIGen cannot be downloaded directly due to permission issues. Please download it manually and place it in the datasets/apigen folder."
wget -O datasets/glaive/glaive-function-calling-v2.json https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json
wget -O datasets/toolace/data.json https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data.json

# Preprocess datasets
python ./scripts/preprocess_apigen_dataset.py
python ./scripts/preprocess_glaive_dataset.py
python ./scripts/preprocess_toolace_dataset.py

# Generate mixed dataset
python ./scripts/mix_datasets.py