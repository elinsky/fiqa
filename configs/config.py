"""Model config in json format"""

CONFIG = {
    'data': {
        'train_path': 'data/task1_headline_ABSA_train.json',
        'test_path': 'data/task1_headline_ABSA_test.json',
        'n_level_1_classes': 4,
        'n_level_2_corporate': 12,
        'n_level_2_economy': 2,
        'n_level_2_market': 4,
        'n_level_2_stock': 9,
        'vocab_size': 1000,
        'seed': 21,
        'buffer_size': 1000,
        'batch_size': 32,
        'train_split': 0.80,
        'validation_split': 0.20,
        'aspect_labels': [['Corporate, Appointment'], ['Corporate, Company Communication'],
                          ['Corporate, Dividend Policy'], ['Corporate, Financial'], ['Corporate, Legal'],
                          ['Corporate, M&A'], ['Corporate, Regulatory'], ['Corporate, Reputation'],
                          ['Corporate, Risks'], ['Corporate, Rumors'], ['Corporate, Sales'], ['Corporate, Strategy'],
                          ['Economy, Central Banks'], ['Economy, Trade'], ['Market, Conditions'], ['Market, Currency'],
                          ['Market, Market'], ['Market, Volatility'], ['Stock, Buyside'], ['Stock, Coverage'],
                          ['Stock, Fundamentals'], ['Stock, Insider Activity'], ['Stock, IPO'], ['Stock, Options'],
                          ['Stock, Price Action'], ['Stock, Signal'], ['Stock, Technical Analysis']]
    },
    'train': {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 1e-3
    },
    'model': {
        'vocab_size': 1000
    },
}
