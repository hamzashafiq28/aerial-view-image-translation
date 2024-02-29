# aerial-view-image-translation

# Overview

This project provides a script for testing a trained image translation model using PyTorch. The script takes input SAR or RGB images, performs inference using the trained model, and saves the translated images (EO, IR, RGB) to a specified results directory.

## Requirements
- Python 3.9
- PyTorch
- torchvision
- numpy
- Pillow

## Installation
1. Clone this repository:
    ```bash
    git clone <repository_url>
    ```
2. Install the required dependencies:
     ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Navigate to the directory containing the script.
2. Run the script with the desired command-line arguments:
    ```css
    python main.py [--model_path MODEL_PATH] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--type TYPE]
    ```
    - `--model_path`: Path to the trained model file (default: 'pretrained/sar2ir/SAR_IR.pth').
    - `--data_dir`: Path to the directory containing the input images (default: 'input/sar2ir').
    - `--results_dir`: Directory to save the resulting translated images (default: 'results/sar2ir').
    - `--type`: Type of input data ('SAREO', 'SARRGB', 'SARIR', 'RGBIR') (default: 'SARIR').


## Notes
- Ensure that the directory paths provided exist and contain the necessary data.
- The script automatically selects the appropriate input domain based on the specified data type.

## Example
```css
python main.py --model_path pretrained/sar2ir/SAR_IR.pth --data_dir input/sar2ir --results_dir results/sar2ir --type SARIR
