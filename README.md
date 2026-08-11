# 🧠 Bayesian Deep Learning — Uncertainty Estimation & OOD Detection

## 📌 Project Overview

This project investigates **uncertainty estimation in deep neural networks** using two approaches:

* **MC-Dropout**
* **Deep Ensembles**

The goal is to move beyond standard point predictions and investigate **how confident a neural network is about its predictions**.

The project evaluates uncertainty for both **regression** and **classification** tasks, including the decomposition of predictive uncertainty into **aleatoric and epistemic uncertainty**.

It also investigates **out-of-distribution (OOD) behaviour** by evaluating Fashion-MNIST-trained classifiers on the **notMNIST** dataset.

---

## 🎯 Objectives

The project investigates:

* How MC-Dropout can approximate Bayesian inference
* How Deep Ensembles can provide predictive uncertainty
* The difference between **aleatoric** and **epistemic** uncertainty
* Whether uncertainty correlates with prediction error
* How well the models are calibrated
* Whether uncertainty can provide a signal for out-of-distribution inputs
* The computational trade-offs between MC-Dropout and Deep Ensembles

---

## 🔬 Methods

### MC-Dropout

MC-Dropout keeps dropout active during inference and performs multiple stochastic forward passes through the same neural network.

In this project:

* **50 stochastic forward passes** were used
* Different dropout masks produce different predictions
* The mean prediction is used as the final prediction
* Variation across stochastic predictions provides an estimate of epistemic uncertainty
* For regression, the model also predicts aleatoric variance

```text
Input
  ↓
Neural Network + Dropout
  ↓
50 Stochastic Forward Passes
  ↓
Multiple Predictions
  ↓
Mean Prediction + Predictive Uncertainty
```

### Deep Ensembles

Deep Ensembles train multiple independently initialized neural networks using the same architecture.

In this project:

* **5 independently initialized models** were trained
* Predictions from the models were combined
* Variation between model predictions provides an estimate of epistemic uncertainty
* For regression, each model also predicts input-dependent variance

```text
                 ┌── Model 1 ──┐
                 ├── Model 2 ──┤
Input ───────────┼── Model 3 ──┼──→ Ensemble Prediction
                 ├── Model 4 ──┤          +
                 └── Model 5 ──┘      Uncertainty
```

---

## ⚖️ MC-Dropout vs. Deep Ensembles

| Aspect                    | MC-Dropout                         | Deep Ensembles                    |
| ------------------------- | ---------------------------------- | --------------------------------- |
| Models trained            | 1                                  | 5                                 |
| Inference passes          | 50                                 | 5                                 |
| Training cost             | Lower                              | ~5× higher                        |
| Inference cost            | Higher                             | Lower                             |
| Source of stochasticity   | Random dropout masks               | Random weight initialization      |
| Epistemic uncertainty     | Variation across stochastic passes | Variation across ensemble members |
| Memory footprint          | Lower                              | Higher                            |
| Inference parallelization | More limited                       | Highly parallelizable             |

MC-Dropout shifts computational cost toward inference, while Deep Ensembles shift more cost toward training and model storage.

---

# 📊 Experiments & Results

## 1. UCI Energy Efficiency — Regression

The regression experiments evaluate predictive performance, likelihood, calibration, and uncertainty.

### Test Set Results

| Metric | MC-Dropout | Deep Ensemble |
| ------ | ---------: | ------------: |
| RMSE   |     2.3853 |    **2.2127** |
| NLL    |    -1.0196 |   **-1.0933** |
| ECE    |     0.0631 |    **0.0419** |

### Training Set Results

| Metric        | MC-Dropout | Deep Ensemble |
| ------------- | ---------: | ------------: |
| Training RMSE |     2.1827 |    **2.0320** |
| Training NLL  |    -1.0656 |   **-1.1390** |

Deep Ensemble achieved better RMSE, NLL, and ECE in the regression experiment.

---

## 2. Regression Uncertainty Analysis

| Metric                        | MC-Dropout | Deep Ensemble |
| ----------------------------- | ---------: | ------------: |
| Mean uncertainty              |     0.6434 |    **0.5626** |
| Mean absolute error           | **0.1780** |        0.1801 |
| Uncertainty-error correlation |     0.6376 |    **0.7204** |

Deep Ensemble produced slightly lower mean uncertainty while achieving a stronger correlation between uncertainty and prediction error.

The report found that uncertainty was higher in regions with larger prediction variability, particularly around extreme heating-load values and transition regions.

The average uncertainty (`2 × std`) reached approximately **1.04** in higher-uncertainty regions compared with approximately **0.36** in lower-uncertainty regions.

---

# 3. Fashion-MNIST — Classification

The classification experiments evaluate predictive accuracy, negative log-likelihood, and calibration.

### Test Set Results

| Metric   | MC-Dropout | Deep Ensemble |
| -------- | ---------: | ------------: |
| Accuracy |     87.55% |    **88.12%** |
| NLL      |     0.3520 |    **0.3366** |
| ECE      |     0.0371 |    **0.0277** |

### Training Set Results

| Metric            | MC-Dropout | Deep Ensemble |
| ----------------- | ---------: | ------------: |
| Training Accuracy |     89.43% |    **89.97%** |
| Training NLL      |     0.3021 |    **0.2794** |

Deep Ensemble achieved slightly higher accuracy and lower NLL and ECE than MC-Dropout.

---

# 4. Classification Uncertainty

The uncertainty analysis investigated whether predictive entropy and confidence were related to prediction correctness.

| Metric                               | MC-Dropout | Deep Ensemble |
| ------------------------------------ | ---------: | ------------: |
| Mean entropy — correct predictions   |     0.3543 |    **0.2998** |
| Mean entropy — incorrect predictions |     1.0059 |    **0.9387** |
| Mean confidence                      |     0.8387 |    **0.8535** |
| Overall accuracy                     |     87.55% |    **88.12%** |
| Calibration gap                      |     0.0368 |    **0.0277** |
| Overall mean entropy                 |     0.4355 |        0.3757 |
| Entropy gap                          |     0.6516 |        0.6389 |

Incorrect predictions showed substantially higher predictive entropy than correct predictions.

This indicates that uncertainty provided a useful signal for identifying predictions that were more likely to be incorrect.

The statistical analysis reported **p < 0.001** for the difference in entropy between correct and incorrect predictions for both approaches.

---

# 5. Calibration Analysis

Calibration was evaluated using reliability diagrams and Expected Calibration Error (ECE).

### Calibration Results

| Metric              | MC-Dropout | Deep Ensemble |
| ------------------- | ---------: | ------------: |
| ECE                 |     0.0371 |    **0.0277** |
| Overconfident bins  |          2 |             1 |
| Underconfident bins |         10 |            11 |

Both models showed relatively small calibration gaps.

Deep Ensemble achieved the lower ECE, indicating better calibration in these experiments.

The analysis also found that both models were predominantly **underconfident rather than overconfident** across the mid-to-high confidence range.

---

# 6. Aleatoric vs. Epistemic Uncertainty

For the regression experiment, predictive uncertainty was decomposed into aleatoric and epistemic components.

| Metric                           | MC-Dropout | Deep Ensemble |
| -------------------------------- | ---------: | ------------: |
| Mean aleatoric uncertainty (std) |   0.277693 |      0.272662 |
| Mean epistemic uncertainty (std) |   0.160979 |      0.066117 |
| Mean total uncertainty (std)     |   0.323618 |      0.281319 |
| Epistemic proportion             |     25.22% |         6.66% |
| Aleatoric proportion             |     74.78% |        93.34% |

Aleatoric uncertainty dominated total uncertainty for both approaches.

MC-Dropout produced a substantially larger epistemic component, while Deep Ensemble attributed a larger proportion of uncertainty to aleatoric variation.

---

## Epistemic Uncertainty Across Prediction Ranges

The epistemic share varied across the predicted heating-load range:

| Prediction range | Epistemic proportion |
| ---------------- | -------------------: |
| Lowest range     |               66.62% |
| Middle range     |     As low as 36.96% |
| Highest range    |               62.27% |

The epistemic share was higher toward the extremes of the prediction range and lower in the middle region.

This was consistent with increased model uncertainty in regions that were less represented by the training data.

---

## Error Magnitude Analysis

| Metric             | High Error Samples — Top 20% | Low Error Samples — Bottom 20% |
| ------------------ | ---------------------------: | -----------------------------: |
| Mean aleatoric std |                     0.422578 |                       0.242076 |
| Mean epistemic std |                     0.206473 |                       0.167368 |
| Mean total std     |                     0.471464 |                       0.295797 |
| Epistemic ratio    |                       43.55% |                         55.63% |

The analysis showed that epistemic uncertainty did not directly correspond to error magnitude in every case.

The report found that epistemic uncertainty was more closely related to the sample's position within the input/data distribution than to prediction error alone.

---

# 🌍 7. Out-of-Distribution Behaviour

Fashion-MNIST-trained classifiers were evaluated on **notMNIST**, a different image domain containing letters rather than clothing.

### OOD Results

| Metric          | MC-Dropout | Deep Ensemble |
| --------------- | ---------: | ------------: |
| Mean confidence |     0.6880 |        0.7458 |
| Mean entropy    |     0.8659 |        0.6792 |
| Confidence std  |     0.2159 |        0.2274 |
| Entropy std     |     0.5056 |        0.5385 |

Compared with in-distribution Fashion-MNIST data, the models showed:

* Lower confidence on notMNIST
* Higher predictive entropy
* A broader uncertainty distribution

MC-Dropout showed a larger relative shift in confidence and entropy in these experiments.

### OOD Limitation

The models did **not** become consistently low-confidence on OOD samples.

A meaningful proportion of notMNIST samples still received confidence above **0.8**.

Therefore, predictive confidence and entropy provide a useful **soft OOD signal**, but they should not be treated as a guaranteed standalone OOD detector in this setup.

---

# 🏆 8. Overall Comparison

| Criterion                     | Better Approach   | Result             |
| ----------------------------- | ----------------- | ------------------ |
| Regression RMSE               | **Deep Ensemble** | 2.2127 vs 2.3853   |
| Regression NLL                | **Deep Ensemble** | -1.0933 vs -1.0196 |
| Regression ECE                | **Deep Ensemble** | 0.0419 vs 0.0631   |
| Classification accuracy       | **Deep Ensemble** | 88.12% vs 87.55%   |
| Classification NLL            | **Deep Ensemble** | 0.3366 vs 0.3520   |
| Classification ECE            | **Deep Ensemble** | 0.0277 vs 0.0371   |
| Uncertainty-error correlation | **Deep Ensemble** | 0.7204 vs 0.6376   |
| Training cost                 | **MC-Dropout**    | 1 model vs 5       |
| Inference passes              | **Deep Ensemble** | 5 vs 50            |
| Memory footprint              | **MC-Dropout**    | 1 model vs 5       |

Overall, **Deep Ensembles performed better across the main predictive and calibration metrics measured in this project**, while MC-Dropout required less training and storage.

The trade-off is therefore between **predictive/calibration performance and computational cost**.

---

# 💡 Key Findings

### Deep Ensembles performed better overall

Deep Ensembles achieved better regression RMSE, NLL and ECE, as well as better classification accuracy, NLL and ECE.

### Uncertainty was informative

Incorrect Fashion-MNIST predictions had substantially higher entropy than correct predictions.

### Aleatoric uncertainty dominated

Aleatoric uncertainty represented the majority of total uncertainty for both approaches.

### Epistemic uncertainty varied across the data distribution

Epistemic uncertainty was higher toward the extremes of the predicted heating-load range.

### OOD uncertainty was useful but imperfect

Both approaches showed lower confidence and higher entropy on notMNIST, but residual overconfidence remained.

### Calibration and accuracy are different

The experiments demonstrate that predictive accuracy alone does not describe whether a model's confidence is reliable.

---

# ⚠️ Limitations

* No systematic hyperparameter search was performed.
* The UCI Energy Efficiency dataset contains approximately **770 samples**, limiting how confidently the regression results can be generalized to other tabular datasets.
* Only **5 ensemble members** were used.
* Neither classifier was specifically trained using an OOD detection mechanism.
* Some OOD samples remained highly confident despite being outside the training distribution.

---

# 🚀 Future Work

Potential extensions include:

* Combining Deep Ensembles with MC-Dropout
* Applying post-hoc calibration methods such as temperature scaling
* Replacing the Fashion-MNIST MLP with a small CNN
* Using epistemic uncertainty to drive an active-learning loop
* Evaluating the methods on larger and more diverse datasets
* Investigating dedicated OOD detection and rejection methods

---

# 📊 Visualizations

The `figures/` directory contains visualizations from the experiments, including:

* Predictive uncertainty
* Regression confidence intervals
* Aleatoric and epistemic uncertainty
* Epistemic uncertainty across prediction ranges
* Classification entropy
* Confidence distributions
* Calibration / reliability diagrams
* OOD confidence distributions
* OOD entropy distributions

---

# 🛠️ Technologies

* **Python**
* **PyTorch**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Jupyter Notebook**

---

# 📂 Repository Structure

```text
Bayesian-Deep-Learning/
│
├── data/
│   └── Dataset files
│
├── figures/
│   └── Experimental visualizations
│
├── bayesian-deep-learning-experiments.ipynb
├── fashion_mnist_loader.py
├── notmnist.py
├── instructions.md
├── .gitignore
└── README.md
```

---

# ▶️ How to Run

The repository includes `instructions.md` with the requirements and setup information for running the assignment.

After installing the required dependencies, open:

```text
bayesian-deep-learning-experiments.ipynb
```

and execute the notebook cells sequentially.

Refer to `instructions.md` for the specific project requirements and setup instructions.

---

# 📚 References

* Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*
* Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.*
