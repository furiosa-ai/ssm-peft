DEVICES=${1:-0}
# train
python run_all.py train.py --device $DEVICES --cfg cfg/exps/cifar/* --override