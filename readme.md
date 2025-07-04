# Parkinson's Disease Diagnosis via Voice Analysis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete implementation of our novel machine learning pipeline for Parkinson's Disease (PD) diagnosis using voice biomarkers. Our approach combines Wasserstein-validated data augmentation, hybrid ensemble learning, and clinical deployment optimization to achieve state-of-the-art performance.

## Key Features

- 💡 **Wasserstein-validated augmentation**: SMOTE + Gaussian noise with quantitative distribution preservation
- 🧠 **Hybrid ensemble**: Combines 7 traditional ML models with novel 1D-CNN for tabular data
- ⚡ **Real-time capability**: 3.6ms inference time for clinical deployment
- 📊 **Statistical rigor**: Bootstrap CIs and McNemar's test for robust validation
- 🏥 **Clinical insights**: Identified spread1 and PPE as key vocal biomarkers

## Performance Highlights

| Metric          | XGBoost | 1D-CNN | Soft Voting |
|-----------------|---------|--------|-------------|
| Accuracy        | 92.3%   | 84.6%  | 89.7%       |
| F1-Score        | 94.5%   | 88.9%  | 92.9%       |
| Inference Time  | 3.6ms   | 5170ms | 8709ms      |
| AUC-ROC         | 95.2%   | 93.4%  | 96.6%       |

## Methodology Overview

1. **Data Acquisition**: UCI Parkinson's dataset (195 samples, 22 features)
2. **Augmentation**: 
   - SMOTE for class balancing
   - Gaussian noise injection (Wasserstein validated)
3. **Feature Engineering**:
   - Correlation-based filtering
   - Recursive Feature Elimination (RFE)
   - PCA dimensionality reduction
4. **Modeling**:
   - 7 traditional ML models (XGBoost, RF, SVM, etc.)
   - Novel 1D-CNN for tabular data
   - Soft voting ensemble
5. **Evaluation**:
   - Bootstrap confidence intervals
   - McNemar's test for statistical significance
   - Clinical efficiency benchmarking

## Installation

### Prerequisites
- Python 3.11
- NVIDIA GPU (recommended for CNN training)

# Download dataset
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
```


## Results Reproduction

1. **Preprocessing and Augmentation**:
```python
python src/preprocessing.py --smote_strategy 0.8 --noise_level 0.05
```

2. **Feature Engineering**:
```python
python src/feature_engineering.py --rfe_features 12 --pca_variance 0.95
```

3. **Model Training**:
```python
python src/models.py --include xgboost 1d_cnn voting
```

4. **Evaluation**:
```python
python src/evaluation.py --bootstrap_iterations 1000 --mcnemar_test
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite using the following format while the paper is under review:

```bibtex
@misc{rizvi2024parkinsons,
  title={Parkinson's Disease Diagnosis via Wasserstein-Validated Voice Augmentation and Hybrid Ensemble Learning},
  author={Rizvi, Shah Mohammad and Akter, Rume and Siyam, Md. Aman Uddin and Shorna, Sumaiya Alam and Dey, Puja},
  year={2025},
  howpublished={GitHub repository},
  url={https://github.com/yourusername/parkinsons-voice-diagnosis}
}
```

For the latest updates on publication status, please check this repository.

## Contact

For questions or collaborations, please contact:

Shah Mohammad Rizvi  
[smri29.ml@gmail.com](mailto:smri29.ml@gmail.com)  
[ORCID: 0009-0005-5413-2396](https://orcid.org/0009-0005-5413-2396)

Department of Computer Science and Engineering  
International University of Business Agriculture and Technology  
Dhaka, Bangladesh
