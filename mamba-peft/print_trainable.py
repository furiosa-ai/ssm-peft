import argparse
from utils.model_utils import TrainableParamsDb

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size")
    parser.add_argument("--peft")
    args = parser.parse_args()

    print(TrainableParamsDb().get_trainable_params(args.size, args.peft))


if __name__ == "__main__":
    main()
