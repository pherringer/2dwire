# Classification of measurement-based quantum wire in stabilizer PEPS

Code to reproduce the numerical lemma in arXiv:2207.00616. 

## Requirements

Python 3.8.5, Numpy, Scipy, Matplotlib, Numba. For a detailed list, see the included .yml file. To create an Anaconda virtual environment directly from this file, use `conda env create -n [name] --file=2dwire_env.yml`. 

## Usage

```
python stabilizers.py
cp stabs5_reps* stabs5_reps.pkl
python throughput.py
```

## Output

The file tp_classes_...png should match Figure 6 in the paper. The file tp_grids_...pkl contains the raw transmission capacity data for all of the $[5,1]$ stabilizer PEPS. To obtain the 13 classes use `np.unique(grids, axis=0)`.