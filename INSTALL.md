# 1. Environment Setup
## 1.1 Setup and activate the Conda environment
```
conda create -n bhar_codes python=3.10
conda activate bhar_codes
```

## 1.2 Install PyTorch (in the conda environment)
```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

## 1.3 Install all other packages (in the conda environment)
Install the required packages from the root directory of the artifact:
```
cd ~/BHAR_codes_artifact
pip install -e .
```
The project's pyproject.toml file specifies the required Python packages and project configuration. This command installs the required dependencies and sets up the artifact as a Python package within the active Conda environment.

# 2. Dataset Download
## 2.1 Create a dataset directory under your home directory
```
cd ~
mkdir codes_artifact_datasets
```
## 2.2 Download datasets
Navigate to the dataset_installation folder from the project root folder, and download the datasets
```
cd ~/BHAR_codes_artifact/dataset_installation
./install_datasets.sh
```
*Make sure the script is executable (chmod +x install_datasets.sh)*

# 3. Installation Complete
This concludes all installation steps needed to run the code in the repository!
- when running code, make sure that the conda environment is active (bhar_codes)






