import pandas as pd
for name in ['sentiment', 'emotion', 'sarcasm']:
    df = pd.read_excel(f'training/datasets/{name}_dataset.xlsx')
    print(f'{name}: {len(df)} samples - {df.label.value_counts().to_dict()}')
