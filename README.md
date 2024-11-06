# Team_SIOL

## Automated Neural Architecture Search for Optimal Model Design

This project leverages Neural Architecture Search (NAS) using AutoKeras to discover the best-performing model architecture for our task automatically. We start with an initial model design phase, utilize NAS to explore and optimize model structures, and then retrain the best-found architecture from scratch to ensure robustness. Finally, we saved the optimized model weights and provided a script to evaluate the model on unseen test data.

### Neural Architecture Search (NAS)

NAS is a technique that aims to discover optimal neural network structures, often using AutoML tools like AutoKeras. The goal is to identify high-performance architectures that outperform manually designed models through a search process that evaluates possible configurations.

The process consists of three main components:

1. **Search Space:** In this project, we utilized the original search space defined in the ```AutoKeras``` library.
2. **Search Strategy:** We employed Bayesian Optimization as the search strategy. 
3. **Performance Estimation Strategy:** We evaluated the best model in terms of validation accuracy. We divided the original training data into training and validation sets. 

### Formulation

Let:

- **A** be the set of all possible neural architectures.
- **f(A, θ)** be the performance function of an architecture $$A$$ with parameters $$θ$$.

The goal of NAS (Neural Architecture Search) is to solve:

$$
A^* = \underset{A \in \mathcal{A}}{\text{argmax}}  \mathbb{E} [f(A, \theta^*)]
$$

where $$θ^*$$ represents the best parameter set for $$A$$ obtained through training.

### Training Objective

Once the optimal architecture $$A^*$$ is identified, it is retrained from scratch. During retraining, we minimize a loss function $$L$$ over the training data $$(x_i, y_i)$$:

$$
\theta^* = \underset{\theta}{\text{argmin}}   L(f(A^*, \theta), y_i)
$$

Common objectives include categorical cross-entropy for classification:

$$
L_{\text{cross-entropy}} = - \sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

where $$\hat{y}_i$$ represents the predicted probability for the correct class.

## Running the NAS Script

To run the NAS on the training dataset, use the following command:

```bash
python nas.py --data_dir path/to/dataset --result_dir path/to/results --model_save_path path/to/best_model.keras --best_weights_file path/to/best_weights.h5
```


## Retraining the best architecture from scratch using training data
To retrain the best model obtained in NAS using the training data, use the following command:

```bash
python train.py --data_dir path/to/dataset --result_dir path/to/results --best_weights_file path/to/results/best_model.weights.h5
```


## Evaluating the best model on test data

To evaluate the trained model on a new test dataset, use the following command:

```bash
python evaluate_model.py --test_dir path/to/test_data --weights_path path/to/best_model.weights.h5
```
