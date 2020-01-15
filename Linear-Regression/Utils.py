import os 

DATASET_PATH = ""
DATASET_URL = ""

def fetch_n_load_dataset(dataset_url = DATASET_URL, dataset_path = DATASET_PATH):
		"""
		Fetches and load dataset.

		:param      dataset_url:   The dataset url
		:type       dataset_url:   { The URl of dataset as variable}
		:param      dataset_path:  The dataset path
		:type       dataset_path:  { PATH details for the dataset as the var}

		:returns:   The file at a particular file location .
		:rtype:     { return_type of this function is the file path }
		"""
		if not os.path.isdir(dataset_path):
			os.makedirs(dataset_path)
		tgz_path = os.path.join(dataset_path,tgz_path)
		csv_path = os.path.join(dataset_path,"FuelConsumptionCo2.csv")
		return pd.read_csv(csv_path)