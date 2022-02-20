import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_tokenizer(dataset_dir: str):
    doc_paths = [os.path.join(dataset_dir, name)
                 for root, dirs, files in os.walk(reuters_path)
                 for name in files]
    tokenizer = Tokenizer(BPE())
    # Without a pre-tokenizer that will split our inputs into words, we might get tokens that overlap several words:
    # for instance we could get an "it is" token since those two words often appear next to each other. Using a
    # pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer.
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=doc_paths, trainer=trainer)
    tokenizer.save("../data/tokenizer-reuters.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str,
                        help="Path to the root of the dataset.")
    args = parser.parse_args()
    reuters_path = args.dataset_dir
    train_tokenizer(reuters_path)
