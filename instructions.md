Instructions

- Python 3.9+
- Conda (Anaconda or Miniconda)

ENVIRONMENT SETUP

1. Create the conda environment:
   conda create -n pa2 python=3.9 -y

2. Activate the environment:
   conda activate pa2

3. Install required packages:
   pip install torch torchvision
   pip install numpy scipy scikit-learn matplotlib pandas
   pip install jupyter ipykernel

4. Register the environment as a Jupyter kernel:
   python -m ipykernel install --user --name pa2 --display-name "Python (pa2)"

5. Launch Jupyter Notebook:
   jupyter notebook

VERIFICATION

To verify all packages are installed correctly:
   python -c "import torch; import numpy; import sklearn; print('All packages installed successfully!')"

