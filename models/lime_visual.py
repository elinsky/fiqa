import numpy as np
import tensorflow as tf
import tokenizers
from lime.lime_text import LimeTextExplainer

from configs.neural_baseline_config import CONFIG
from utils.config import Config


def create_visual(model: tf.keras.Model, tokenizer: tokenizers.Tokenizer, sentence: str, label: tf.Tensor):
    # Label needs to be a one-hot encoded tensor for the sentence's correct aspect label
    # Writes output to lime_visual.html

    # Create explainer
    config = Config.from_json(CONFIG)
    class_names = config.data.aspect_labels
    class_names = [class_name[0] for class_name in class_names]
    explainer = LimeTextExplainer(class_names=class_names, random_state=21)

    def predict_proba(sentence: str) -> np.ndarray:
        inputs = tokenizer(sentence, return_tensors="tf", padding='max_length', max_length=len(sentence))
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        inputs = input_ids, token_type_ids, attention_mask
        outputs = model(inputs)
        sentiment_logits, aspect_prob = outputs
        aspect_prob = aspect_prob.numpy()
        return aspect_prob

    # Explain instance
    label_position = int(tf.where(label))
    exp = explainer.explain_instance(sentence, predict_proba, num_features=20,
                                     labels=[label_position])  # Labels should be the correct label.
    # Write visual to html file
    exp.save_to_file('lime_visual.html')
