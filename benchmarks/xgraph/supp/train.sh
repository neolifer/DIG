#!/bin/sh
#for i in "BA_Community"
#do
#  if test $i == "BA_shapes" || test $i == "BA_Community"
#  then
##    python graphmask_explain.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 20 --use_baseline True
##    python graphmask_explain_single.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 20
##    python gnnexplainer_explain1.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 20
#    python pgexplainer_explain.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 20
#  else
##    python graphmask_explain.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 40 --use_baseline True
##    python graphmask_explain_single.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 40
##    python gnnexplainer_explain1.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 40
#    python pgexplainer_explain.py --dataset_name $i --model_name GM_GCN_100 --num_layers 3 --dim_hidden 40
#  fi
#done
for i in "Cora"
do
  python graphmask_explain_single.py --dataset_name $i --model_name GM_GCN_nopre --num_layers 2 --dim_hidden 64
#  python pgexplainer_explain.py --dataset_name $i --model_name GM_GCN_nopre --num_layers 2 --dim_hidden 64
done
