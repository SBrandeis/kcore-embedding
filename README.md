# k-core-embedding 

Our paper can be found [here](https://github.com/SBrandeis/k-core-embedding/blob/master/ABOUT%20GRAPH%20DEGENERACY%2C%20REPRESENTATION%20LEARNING%20AND%20SCALABILITY%20.pdf)

# How to use 

After cloning, create your virtual environment, then :

```
cd k-core-embedding/
```

**Running multiple experiments**

```
pip install -r requirements.txt
python setup.py install
sh multi_exp.sh <graph>
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
