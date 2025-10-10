# Evons data - download instructions

Codes expect data to be in this folder.

Due to the large size of the data, it was not possible to upload it directly to GitHub. Below are instructions for downloading it from other hosts:

- The complete Evons dataset in CSV format can be downloaded following curators' instructions here: https://github.com/krstovski/evons

- precomputed embeddings can be accessed by [clicking here](https://drive.google.com/drive/folders/1X27WjPEKzAcC5jXai8cuI8FEmWkjJb6l?usp=sharing).

- If you prefer, you can compute embeddings by yourself: BERT embeddings will be created automatically on-the-fly when executing notebooks, while Mistral embeddings can be computed using `evons/data_preprocessing/create_embeddings_mistral.py` script.