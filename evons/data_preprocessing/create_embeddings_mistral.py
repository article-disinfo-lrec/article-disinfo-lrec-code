import os
import torch
import pandas as pd
from mistralai import Mistral
from tqdm import tqdm

# ========= Configuration =========
API_KEY = 'YOUR-KEY-HERE'
MODEL_NAME = 'mistral-embed'
BATCH_SIZE = 256

# Relative paths (script expected in evons/data_preprocessing)
DATA_CSV_PATH = '../data/evons.csv'  # place evons.csv inside evons/data/
TITLE_OUT_PATH = '../data/title_embeddings_mistral.pt'
DESC_OUT_PATH = '../data/desc_embeddings_mistral.pt'

# ========= Helper =========
def compute_embeddings(client, texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = [str(t) for t in texts[i:i+batch_size]]
        if not batch_texts:
            continue
        resp = client.embeddings.create(model=MODEL_NAME, inputs=batch_texts)
        batch_embs = [x.embedding for x in resp.data]
        embeddings.extend(batch_embs)
    return embeddings


def main():
    if API_KEY == 'YOUR-KEY-HERE':
        raise ValueError('Edit API_KEY.')

    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(f'Missing data file at {DATA_CSV_PATH}. Place evons.csv there.')

    df = pd.read_csv(DATA_CSV_PATH)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')

    titles = df['title'].tolist()
    descs = df['description'].tolist()

    client = Mistral(api_key=API_KEY)

    print('Computing title embeddings...')
    title_embeddings = compute_embeddings(client, titles)
    torch.save(title_embeddings, TITLE_OUT_PATH)

    print('Computing description embeddings...')
    desc_embeddings = compute_embeddings(client, descs)
    torch.save(desc_embeddings, DESC_OUT_PATH)

    print('Done.')


if __name__ == '__main__':
    main()
