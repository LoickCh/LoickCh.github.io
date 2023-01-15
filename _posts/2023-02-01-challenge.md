---
layout: post
title:  Kaggle TPS.Nov.2022
date:   2023-01-02
description: Tabular Playground Series - November 2022
tags: Kaggle
categories: challenge
---

# TPS Nov-2022

Hello, in this note I explain my personnal solution to the TPS challenge where I ranked 18/689 (2.6 %). 

<p> <br> </p>
### Table of content

* Subject
* Approach
* Other solutions
* Conclusion

<p> <br> </p>

## 1.Subject

In this competition, there is a collection of predictions and ground truth labels for a binary classification task we do not know about. The objective is to use blending techniques to improve the model's predictions by combining submissions. Stacking and blending are machine learning techniques learning a meta-model:
- stacking: trains a meta-model from a pool of base models. The base models are trained on a complete training set, then the meta-model is trained on the features that are outputs of the base models. This operation can be repeated several times, at the risk of overfitting and having a more complex model.
- blending: trains a meta-model on a separate holdout set. It is very similar to stacking however, it does not use the same data.

About the data format, submission files are divided in two parts. In the submissions folder, the first half of the rows contains training set predictions. Their name corresponds to the log-loss over the training set. The other half contains predictions for the testing set. The testing set consists of 20,000 rows, and the submission expects probabilities.


<p> <br> </p>

## 2. Approach

I am inspired by several notebooks and ideas from other participants. In this competition, exploratory data augmentation was not crucial so I did not focus on it. However, preparing data and chose a good training pipeline helps me to achieve my results.

### 2.1 Pre-processing

#### 2.1.1 Data-format

Instead of using a folder of submission files, we concatenate all of them in a single dataframe object containing 5k columns and 40k rows where each column is a submission and each row is an index. Then, to accelerate the loading, we convert the concatenated dataset to a binary format. In our case we select the feather (.ft) extension, but many exists: pickle (specific to Python), parquet, etc. 

```python
X=pd.read_feather('../input/tps2022novfeather/train.ftr')
y=pd.read_csv('../input/tps2022novfeather/train_labels.csv')
```

Once loaded, we observed that some inputs are above 1 or below 0. Since it is supposed to contain probabilities, we clip values between 1e-7 (close to minimum float32 precision) and 1-1e-7 to ensure we manipulate probabilities. Alternatively, we could have flag set a value for outliers (for instance -1 below and 2 above) and use randomforest-like algorithms.

```python
# Clip data
X = X.clip(EPS, 1-EPS)
```

After that, we split our dataset in training and testing datasets.

```python
# Copy and split datasets in two sets.
X, X_test = deepcopy(X.iloc[:TRAIN_SIZE,:]), deepcopy(X.iloc[TRAIN_SIZE:,:])
y, y_test = deepcopy(y.iloc[:TRAIN_SIZE]), deepcopy(y.iloc[TRAIN_SIZE:])
```

Then, before to train models, we modify our datasets according to a bag of tricks coming from diverse sources.

#### 2.1.2 Bag of tricks

Since the submissions contain binary probabilities, we can consider whether it would be better to invert some of these probabilities in order to improve the log-loss score. We found that 9 elements had better log-loss scores when their probabilities were inverted, so we made this change.

```python
better_invert_proba_idx = np.where(
    np.array(
        [
            (log_loss(y, p, eps=EPS) - log_loss(y, 1 - p, eps=EPS)) 
        for p in X.T
        ]
        )> 0
    )[0]
X.iloc[better_invert_proba_idx,:] = 1 - X.iloc[better_invert_proba_idx,:]
```

Then, we used logits instead of probabilities, even though it is generally not necessary or advisable to convert predicted probabilities into logits before solving a binary classification problem. This step is optional, but allows for better results when learning logistic regression according to [this](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/364013).

I only kept these two ideas in my final submission but many have been tested and discussed by the community. For instance, considering bad classifier are nearly random classifiers, they have a ROC-AUC scores close to 0.5. Other pre-processing strategies tested were to calibrate the probabilities using isotonic regression or an embedding of the inputs with a neural network.

### 2.2 Model

Like many other candidates, I used LightAutoML, a Python library performing automated machine learning with high GPU compatibility. There are several other automated machine learning libraries available, such as H2O, PyCaret, TPOT, and Autosklearn. Of these, only [LightAutoML](https://github.com/sb-ai-lab/LightAutoML) and [TPOT](https://github.com/EpistasisLab/tpot) produced good results for me. I didn't spend much time on tuning the hyperparameters, so it's possible that some of the other libraries may perform just as well or better if given more time and attention.

#### 2.2.1 Feature selection

Feature selection is an important step in building a model. Reducing the number of features often reduces overfitting and speeds up training, especially with 5,000 columns. Ideally, the feature selection method should be independent of the model and should discriminate the data by itself. For instance, we can select features having a high variance or remove features highly correlated to ensure diversity and erase redundancy (for instance using hierarchical clustering on the Spearman rank-order correlations or the Jensen Shanon distance, [cf](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/368905)).

Another approach consists on training a model to predict labels and then retrieve the most important features for predictions. It is even more effective when training algorithms with L1 or L2 regularisations, as they penalise the model useless model weights. Alternatively to feature selection, we could have applied dimensionality reduction techniques such as Principal Component Analysis or Partial Least Square Regression. It is another way to reduce the shape of our input data. In this competition, PCA, or neural network encoding do not lead to the best results, so I did not keep them.

Since performing feature selection by training gave me the best results, I chose it even though it is more prone to overfitting. I tried two different libraries: scikit-learn (in combination with the catboost and LinearSVC algorithms) and LightAutoML set with few parameters. The final submission uses LightAutoML and more specifically the `TabularAutoML` class. My method is rather similar to the one developped by [A.Ryzkhov](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/363483). First of all, we need to specify a task. The subject of the challenge is a blending problem, but seen from another angle, it is nothing more or less than a binary classification where each column is a submission. Then, we chose the selection_params:
- importance_type: we have the choice between permutation (calls `NpPermutationImportanceEstimator`) or gain (calls `ModelBasedImportanceEstimator`). Permutation uses a random permutation of elements in a single column for each feature to calculate its significance. Intuitively, if an element is significant, performance deteriorates when it is shuffled. A more advanced techniques compares the importance of features fitted to the target against the importance of features when fitted to noise defined as the shuffled target ([cf.](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances)). 
- feature_group_size: specify the number of elements permuted during each step of the search. The less it is, the more accurate we are, but the longer it is.
- select_algos: `linear_l2` corresponds to LBFGS L2 regression based on torch while `gbm` corresponds to gradient boosting algorithms from LightGBM library. There are several others, but LightGBM is a good starting point because it allows for L1 and L2 regularisation, thus eliminating unnecessary features. 
- mode: specify the feature selection mode between no selection (0), drop zero importances (1) and iterative permutation importances (2).

```python
automl = TabularAutoML(
    task = Task('binary', metric = 'logloss'), 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    selection_params = {
        'importance_type': 'permutation',
        'fit_on_holdout': True,
        'mode': 2, 
        'feature_group_size': 50,
        'select_algos': ['linear_l2','gbm']
    },
    reader_params = {'n_jobs': N_THREADS}
)
```

The feature selection takes around 1h on Kaggle. 

#### 2.2.2 Training

I have trained and finetuned lightgbm classifiers, catboost classifiers and scikit-learn algorithms (HistGradientBoostingClassifier, LogisticRegression, etc.) using Optuna and even pure deep learning networks, but LightAutoML gave me the best results.

The training loop relies again on the `TabularAutoML` class with few other hyperparameters:
- general_params: specifies which algorithms trained.
- nn_params: specifies the neural network parameters. The neural architecture is predifined and flagged with `mlp`, `dense`, `resnet` or `denselight` but we can custom it changing the activation function (or implementing new ones such as [Growing Cosine Unit](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/366518), MISH), clipping gradients, select the batch size, etc. 

```python
# General neural network parameters
general_nn_params = {
    "random_state":SEED, 
    "act_fun": Mish, 
    "n_epochs": 100, 
    "bs": 2048, 
    "num_workers": 0,
    "path_to_save": None,
    "clip_grad": True, 
    "clip_grad_params": {"max_norm": 1}, 
    "verbose": None,
    "pin_memory":True,
}

# Main class
automl = TabularAutoML(
    task = Task('binary', metric='logloss'), 
    cpu_limit = N_THREADS,
    general_params = {
        "use_algos": [["linear_l2", "mlp", "denselight","lgb"]], 
    },
    nn_pipeline_params = {
        "use_qnt": False
    },
    nn_params={
        "0": {**general_nn_params},
        "1": {**general_nn_params, 
              **{
                    'bs': 512,
                    'opt_params': {'lr': 0.04047, 'weight_decay': 2.43e-05}, 
                    'clip_grad': True, 
                    'clip_grad_params': {'max_norm': 0.0979877026847337}
                }
             },
        "2": {**general_nn_params, 
              **{
                    'bs': 64, 
                    'opt_params': {'lr': 0.00543,'weight_decay': 2.2282e-05},
                    'clip_grad': True, 
                    'clip_grad_params': {'max_norm': 4.683639733744027}
                }
            }
    },
    tuning_params={
        "max_tuning_time": 3600 * 10,
        "fit_on_holdout": True
    },
    reader_params = {
        'n_jobs': N_THREADS
    },
)

# Training
X['label']=y['label']
oof_pred = automl.fit_predict(
    X, 
    roles={'target': 'label'}, 
    cv_iter=train_val_indices,
    verbose=3
)
```

Once the model is trained, we simply predict the probabilities on the test set. If we have many submissions, a last step consists on averaging submissions. A good averaging strategy is to find the optimal weights for each submission on each k-fold sets. It can be done using `scipy.minimize` or even applied the whole pipeline previously developed, but the snake bites the tail at the risk of overfitting.

## 3. Other solutions

Some winner candidates have shared their solutions:

### [1.st](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/369674) place solution

Keypoints:
- Framework: LightAutoML
- Data: use logits instead of probabilities.
- Architecture: XGBoost, LightGBM, NNs (with MISH and Swish activation functions).
- Training: 10 k-Fold cross-validations. It seems to be good to increase the number of folds while doing cross-validation. However, if we increase it too much, we can overfit on our data.
- Ensembling: uses [`scipy.minimize`](https://www.kaggle.com/code/pourchot/stacking-with-scipy-minimize) to find the optimal weights on the OOF.

### 3.rd place solution

Keypoints:
- Framework: AutoGluon.
- Feature selection: drop features containing values out of range (0,1) and using Spearman correlation.
- Calibrations: uses Isotonic Calibration.

### [7.th](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/369731) place solution

Keypoints:
- Feature selection: drop features based on the difference between the loss value of the model with this feature and without it. Performs feature selection with [null-importance](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances).

## 4. Conclusion

This toy competition was original in a sense it consists on ensembling models to get better performances even if we tackle it as a binary classification task. In fact, we build a meta-model for a task we do not know about. Interestingly, auto machine learning frameworks work well and gave the best results. Many participants have used LightAutoML pushed by simple and effective notebooks written by the authors of the library. In this challenge, the exploratory data analysis was not so important, but having a good feature selection algorithm helped a lot.

----------------------------------------------------

Other sources:
- [Blending and stacking](https://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413c)
- [Calibration](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/364013)

Interesting techniques not used:
- [OOF Forward selection](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/364834)