# Traffic Non-recurrent Congestion Classification


* * *


## 1. Data Preprocessing

Before training the model, the dataset should be generated with preprocessing and train/test split.
The dataset is generated in the route of *'data/{dataset_name}'*.

1. Filter the cascading incidents as a csv file *./data/accident_all.csv*.

    ```python 0. Cascading_incident_filtering.py```


2. Run the data preprocessing file with the incident road sids and dataset options. There are several options for data preprocessing according to the congestion definition, which is explained in the below.  

    ```bash data_preprocessing.sh```  



### a. Traffic Congestion Definition

While filtering the congestion status data, we used the congestion conditions for speed dataset. The explanations for each dataset is as follows.

- **mtsc** : traffic congestion status if more than 50% of subgraph roads are congested / 1 hour speed data
(1. data_preprocessing_mtsc.py)

- **mprofile** : traffic congestion status if incident road & more than 50% of 1hop incoming roads are congested / 2 hours speed data
(3. data_preprocessing_mprofile.py)

- **mprofile2** : also consider incident reports which occurred on other roads in subgraph
(3. data_preprocessing_mprofile2.py)


### b. Train/Test split

The train/valid/test set is split according to the period. Each dataset includes 3 months for train set and 1 month each for valid and test set.  


* * *


## 2. Non-recurrent Congestion Classification

### Method
- Semi-supervised Anomaly Detection
- DeepSAD + OCGNN


### Implementation
```cd NRC_classification```

```bash run_SAD{incident number}.sh```

There are five number of incident cases. The name of dataset is constructed as {accident_sid}_{dataset_option}.  
 

* * *


## 3. Baselines

### Implementation
```cd NRC_classification```

```python baseline.py --target-sid {incident road sid} --data-type {data_preprocessing_option}```

The dataset name should be changed before the running the model.

The result csv file is saved in *'result/{dataset name}_baselines.csv'* where dataset name is *{target_sid}_{data_type}*.

