![image](https://github.com/user-attachments/assets/336681d3-6df2-4d2f-86ec-f1ba19c47c65)


# Rock Joint Sets Analyzer

This Python module provides a flexible workflow for analyzing geological joint sets from dip-direction/dip measurements. By integrating stereographic projection and clustering algorithms from `mplstereonet`and `kmedoids` packages, it enhances the estimation of joint set families — offering greater control and adaptability in the analysis process. Visualization is handled using the `mplstereonet` package, which supports stereonet plotting of planes, poles, and cluster centroids.

Fernando García Bastante<br>
Universidade de Vigo<br>
For Educational Purposes


## 📌 Features

- 🧭 **Geological Data Input**: Reads dip-direction and dip angle pairs from plain text files.
- 📊 **Clustering**: Uses k-means and optionally k-medoids to identify joint set groupings.
- 📐 **Centroid Calculation**: Computes the centroid (average orientation) of each joint set.
- 📈 **Stereonet Visualization**:
  - Poles to planes
  - Cluster centroids
  - Density contours
  

## 🔧 Installation

### Step-by-step Setup

1. **Create a new Python environment** (recommended):
```bash
conda create -n sets_env
conda activate sets_env
```

2. **Install required dependencies**:
```bash
conda install numpy matplotlib mplstereonet kmedoids
```

3. **Download the module**:

Simply download and save the `rock_joint_sets.py` file in your working directory. You can then import it directly in your scripts or notebooks.


## 📁 Data Input Format

The input should be a plain `.txt` file with each line containing a dip direction and dip angle, separated by whitespace or a custom delimiter:

```
120 45
133 50
147 42
...
```

## 🗂️ Example Use Case

[Please open and run the example file: rock_join_sets_use.ipynb]

```python
import rock_joint_sets as rjs

# Set file and desired number of clusters
data_file = "my_set.txt"
num = 3

# Run analysis
rjs.run_example(data_file, num)
```

Or more explicitly:

```python
data = rjs.read_data("my_set.txt", delimit="")
strike_cent, dip_cent = rjs.centroids_cal(data, num)
rjs.draw_data_centroids(data, strike_cent, dip_cent)
```



## 📊 Output

- **Plots**: Stereonet plots with:
  - Poles of input measurements
  - Centroid planes
  - Cluster visualization
- **Numerical Data**: Strike and dip of the centroids printed or returned.


## 🙏 Acknowledgements

This project would not be possible without the excellent open-source tools it builds upon:

- **[mplstereonet](https://github.com/joferkington/mplstereonet)**: A powerful Python library for plotting and calculating stereographic projections, essential for structural geology workflows. It is the backbone of the stereonet visualizations and performs critical geometric computations.

- **[kmedoids](https://github.com/kno10/python-kmedoids)**: A flexible implementation of the k-medoids clustering algorithm in Python. It allows more robust identification of joint set clusters by minimizing dissimilarities using medoid representatives.

We are deeply grateful to the authors and maintainers of these libraries for their contributions to the scientific Python ecosystem.
