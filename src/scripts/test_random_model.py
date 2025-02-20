from src.scripts.test_best_model import train_and_test
from src.modeling.models.mlp_rnd import MLP_rnd
if __name__ == '__main__':
    train_and_test(MLP_rnd, 'random_model')