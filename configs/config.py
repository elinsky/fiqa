"""Model config in json format"""

CONFIG = {
    'data': {
        'path': 'data',
        'n_level_1_classes': 4,
        'n_level_2_corporate': 12,
        'n_level_2_economy': 2,
        'n_level_2_market': 4,
        'n_level_2_stock': 9
    },
    'train': {
        'batch_size': 32,
        'epochs': 10
    },
    'model': {
        'vocab_size': 3000
    }
}