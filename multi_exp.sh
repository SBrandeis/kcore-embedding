cd scripts
graph=$1
for config in ./$graph/sample_config_*.json
do
  for params in ./$graph/default_params_*.json
  do
    echo "************** Processing $graph with config : $config and params : $params **************"
    python exec_pipeline.py --config $config --params $params --sub-embedder-params $params
  done
done
