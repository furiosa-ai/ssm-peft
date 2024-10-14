DEVICES=${1:-0}
# train
python run_all.py train.py --device $DEVICES --cfg cfg/exps/spider14b/* --override
# eval
python run_all.py eval.py --device $DEVICES --eval_cfg cfg/eval/spider14b/* --override