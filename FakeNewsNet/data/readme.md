# Data

All code expects the data to be stored in this folder. Due to their size and Twitter/X privacy rules, data cannot be uploaded to GitHub. Use the following sources and steps:

- Raw data
	- Check the original FakeNewsNet repository for the dataset and collection instructions: https://github.com/KaiDMML/FakeNewsNet

- Processed data (anonymized propagation paths)
	- Download the anonymized data from: https://drive.proton.me/urls/6ZDQPJY178#XMn79ZYqEcxt
	- Place the downloaded files directly in this folder. Expected filenames:
		- `anonymized_fake_propagation_paths.jsonl`
		- `anonymized_real_propagation_paths.jsonl`
	- Then use `rebuild_dataset.py` in `data_preprocessing` folder to retrieve tweet-level information through the Twitter/X API and rebuild the dataset locally. Execute `rebuild_dataset.py` file from its own folder.

Note: You will need valid Twitter/X API credentials, and retrieval may be subject to rate limits.