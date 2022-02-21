import argparse
import os

import tokenizers
from tokenizers import Tokenizer, normalizers, pre_tokenizers, trainers, processors, decoders
from transformers import BertTokenizerFast


def train_tokenizer(dataset_dir: str, target_dir: str):
    # https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
    # Tokenizers in the tokenizer library follow the following high level structure:
    # 1) Normalization
    # 2) Pre-Tokenization
    # 3) Model
    # 4) Post-Processing

    doc_paths = [os.path.join(dataset_dir, name)
                 for root, dirs, files in os.walk(dataset_dir)
                 for name in files]

    # Create tokenizer with an EMPTY WordPiece model
    tokenizer = Tokenizer(tokenizers.models.WordPiece(unl_token="[UNK]"))

    # Normalizes raw text. Cleans text by removing control characters and replacing all whitespace characters with
    # the classic one. Converts to lowercase. Applies Normalization Form D (NFD) Unicode normalization.
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )

    # Pre-tokenization splits the text into smaller objects. These are roughly words. The final tokens will either be
    # these words, or sub-words. Without a pre-tokenizer, we might get tokens that overlap several words. E.g. if the
    # words 'it is' appear frequently together, that could end up its own token.
    # Split tokens on spaces and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), pre_tokenizers.Whitespace()])

    # Define special tokens.
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    # Train word piece tokenizer model on our custom dataset
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train(files=doc_paths, trainer=trainer)

    # Define the post-processor.
    # After tokenizing the sentence, we need to add in the CLS and SEP tokens that BERT expects. Add CLS token at the
    # beginning and the SEP token at the end (for single sentences) or several SEP tokens (for pairs of sentences).
    # Start by getting the IDs of the CLS and SEP tokens
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    # Create post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )

    # Create decoder for tokenizer
    # The decoder is the API for converting IDs back into text, removing all special tokens (e.g. CLS), and joining the
    # tokens with spaces.
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    # Wrap the tokenizers.models.WordPiece object inside a transformers.BertTokenizerFast object. Without this step we
    # won't be able to instantiate our pre-trained tokenizer and use it with a transformer model from the transformers
    # library.
    wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)

    # Save tokenizer to disk. Re-load with transformers.BertTokenizerFast.from_pretrained(tokenizer_directory)
    wrapped_tokenizer.save_pretrained(target_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str,
                        help="Path to the root of the dataset.")
    parser.add_argument("target_dir", type=str,
                        help="Path to the directory to save the trained tokenizer.")
    args = parser.parse_args()
    dataset_path = args.dataset_dir
    target_path = args.target_dir
    train_tokenizer(dataset_path, target_path)
