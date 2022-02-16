from configs.config import CONFIG
from models.neural_baseline import NeuralBaseline


def run():
    """Builds model, loads data, trains and validates"""
    model = NeuralBaseline(CONFIG)
    model.load_data()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
