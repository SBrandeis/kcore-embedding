# k-core-embedding 

# How to use 

After cloning, create your virtual environment, then :

**Running multiple experiments**

```
pip install -r requirements.txt
python setup.py install
sh multi_exp.sh
```

**Running single experiment**


```
pip install -r requirements.txt
python setup.py install
cd src
python exec_pipeline.py --config <exp_config> --params <embedding or propag params> --sub-embedder-params <sub embedder params>
```

The project is organized as follows

```
k-core-embedding
|--- data
    |--- xxx.glm 
    |--- xxx.glm    
|--- src Library root
    |--- kce
        |--- embedders  Embedding Methods
        |--- frameworks Propagation Methods
|--- scripts To run experiments
    |--- exec_pipeline.py
    |--- configs.json
|--- output
    |--- exp1
        |--- embeddings
        |--- base_metrics.csv
        |--- configs.json
```