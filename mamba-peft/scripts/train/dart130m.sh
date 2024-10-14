DEVICES=${1:-0}
# train
python run_all.py train.py --device $DEVICES --cfg cfg/exps/dart/* --override
# eval
python run_all.py eval.py --device $DEVICES --eval_cfg cfg/eval/dart/* --override