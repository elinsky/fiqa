from configs.config import CONFIG
from models.naive_baseline import NaiveBaseline


def run():
    """Builds model, loads data, trains and validates"""
    model = NaiveBaseline(CONFIG)
    model.load_datasets()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
