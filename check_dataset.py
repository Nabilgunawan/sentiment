import sys; sys.path.insert(0,'.')
from training.data_utils import load_training_data, validate_labels
from collections import Counter

texts, labels = load_training_data('training/datasets/sentiment_dataset.csv', 'text', 'label')
print(f'Loaded: {len(texts)} samples')

valid_labels = ['Positive', 'Neutral', 'Negative']
ok, invalid = validate_labels(labels, valid_labels)
print(f'All labels valid: {ok}')
if not ok:
    print(f'Invalid: {invalid}')

c = Counter(labels)
print(f'Positive: {c.get("Positive", 0)}, Neutral: {c.get("Neutral", 0)}, Negative: {c.get("Negative", 0)}')
print('Dataset READY for upload to /training')
