DEVICES=${1:-0}
sh scripts/train/spider14b.sh "$DEVICES"
sh scripts/train/spider14b_fromlora.sh "$DEVICES"
sh scripts/train/spider28b.sh "$DEVICES"
sh scripts/train/dart130m.sh "$DEVICES"
sh scripts/train/glu130m.sh "$DEVICES"
sh scripts/train/cifar130m.sh "$DEVICES"