<h1 align='center'>
  Explaining deep learning model for predicting future bridge conditions
</h1>

- **Big idea** -- Various machine learning models are being used to rely on understanding the performance of bridges. However, these machine learning models are difficult to explain and interpret.
- **Small idea** -- Especially, in understanding the insights offered by the complex machine learning models.
- **Birds eye view of the idea** -- We plan to evaluate the performance of the machine learning model explain the relationship of between factors from the model's perspective that is stored with in the model with respect to the data. 
- **Technical details** --  Train a dataset on a variety of commonly used machine learning models for prediction. And, also evaluate the interpretability of the models.

### 🎯 Objective
- The objective of this research study is to evaluate various machine learning model from perspective of accuracy and interpretations.
* We apply these machine learning models to National Bridge Inventory dataset.

### 💪 Challenge
- In the real world dataset, it is a **challenge** to identify patterns and various factors that interact in explaining the decisions made for specific instance.
- Each machine learning model have strengths and weakness that can highlight the patterns within dataset. Therefore, machine learning model can have different performance based on the dataset.

### 🧪 Solution
- We did a comprehensive study of evaluating most commonly evaulated machine learning model from the prespective of predicitive performance and interpretation of the machine learning results.  

### 🎬 Getting started
The following are the steps to setup this project:

#### Download dataset

```zsh
https://drive.google.com/drive/folders/1BcyIwDW6jmljeEygNghaNcv7LD2WxoJm?usp=drive_link
```

####  Clone
```zsh
git clone https://github.com/kaleoyster/ml-nbi.git
```

#### Run requirements.txt

```zsh
pip install -r requirements.txt
```

#### Run the machine learning model

```zsh
cd src
python3 dl.py
```
