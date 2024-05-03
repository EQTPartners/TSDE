import tarfile
import zipfile
import sys
import os
import wget
import requests
import pandas as pd
import pickle
import gdown

# Define a root URL for downloading forecasting datasets used in Salinas et al. https://proceedings.neurips.cc/paper/2019/file/0b105cf1504c4e241fcc6d519ea962fb-Paper.pdf
root = "https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/"


def get_confirm_token(response):
    """
    Extract the confirmation token from the response cookies, which is required for some downloads.
    
    Parameters:
    - response: The response object from a requests session.
    
    Returns:
    - The confirmation token if found; otherwise, None.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    """
    Saves the content of a response object to a file in chunks, which is useful for large downloads.
    
    Parameters:
    - response: The response object from a requests session.
    - destination: The file path where the content will be saved.
    """

    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(url, destination):
    """
    Downloads a file from Google Drive by handling the confirmation token and saving the response content.
    
    Parameters:
    - url: The Google Drive file URL.
    - destination: The file path where the downloaded data will be saved.
    """
    session = requests.Session()

    response = session.get(url, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'confirm' : token }
        response = session.get(url, params = params, stream = True)
    save_response_content(response, destination)   
    
# Ensure the data directory exists
os.makedirs("data/", exist_ok=True)

# Download and extract datasets based on the command-line argument
if sys.argv[1] == "physio":
    # Install PhysioNet dataset 

    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    url_outcomes = "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download"
    os.makedirs("data/physio", exist_ok=True)
    wget.download(url, out="data/")
    wget.download(url_outcomes, out="data/physio")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")
        print(f"Downloaded data/physio")
    os.remove("data/set-a.tar.gz")


elif sys.argv[1] == "pm25":
    # Install PM2.5 dataset 

    url = "https://www.microsoft.com/en-us/research/uploads/prod/2016/06/STMVL-Release.zip"

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    filename = "data/STMVL-Release.zip"
    with requests.get(url, stream=True, headers=headers) as response:
    # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                # Write the content of the file to the local file
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")

    os.remove(filename)
        
    def create_normalizer_pm25():
        """
        Normalizes the PM2.5 dataset by calculating and saving the mean and standard deviation, excluding test months.
        
        
        It saves these values for use in normalizing the training and testing data.
        """
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime", # Use the datetime column as the index
            parse_dates=True,
        )
        
        # Define the months to exclude from the normalization process (test months)
        test_month = [3, 6, 9, 12]
        # Exclude the test months from the dataset
        for i in test_month:
            df = df[df.index.month != i]
        # Calculate the mean and standard deviation for the dataset after excluding the test months
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values

        # Save the mean and standard deviation to a file using pickle
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)
    create_normalizer_pm25()

elif sys.argv[1] == "electricity":
    # Install Electricity dataset 
    
    url = root + "electricity_nips.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/electricity_nips.tar.gz", "r:gz") as t:
        t.extractall(path="data/electricity")
        print(f"Downloaded data/electricity")
    os.remove("data/electricity_nips.tar.gz")

elif sys.argv[1] == "solar":
    # Install Solar dataset 

    url = root + "solar_nips.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/solar_nips.tar.gz", "r:gz") as t:
        t.extractall(path="data/solar")
        print(f"Downloaded data/solar")
    os.remove("data/solar_nips.tar.gz")
    os.rename("data/solar/solar_nips/train/train.json", "data/solar/solar_nips/train/data.json")
    os.rename("data/solar/solar_nips/test/test.json", "data/solar/solar_nips/test/data.json")

elif sys.argv[1] == "traffic":
    # Install Traffic dataset 

    url = root + "traffic_nips.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/traffic_nips.tar.gz", "r:gz") as t:
        t.extractall(path="data/traffic")
        print(f"Downloaded data/traffic")
    os.remove("data/traffic_nips.tar.gz")

elif sys.argv[1] == "taxi":
    # Install Taxi dataset 

    url = root + "taxi_30min.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/taxi_30min.tar.gz", "r:gz") as t:
        t.extractall(path="data/taxi")
        print(f"Downloaded data/taxi")
    os.remove("data/taxi_30min.tar.gz")
    os.rename("data/taxi/taxi_30min/", "data/taxi/taxi_nips")
    os.rename("data/taxi/taxi_nips/train/train.json", "data/taxi/taxi_nips/train/data.json")
    os.rename("data/taxi/taxi_nips/test/test.json", "data/taxi/taxi_nips/test/data.json")


elif sys.argv[1] == "wiki":
    # Install Wiki dataset 

    url = "https://github.com/awslabs/gluonts/raw/1553651ca1fca63a16e012b8927bd9ce72b8e79e/datasets/wiki-rolling_nips.tar.gz"
    wget.download(url, out="data")
    with tarfile.open("data/wiki-rolling_nips.tar.gz", "r:gz") as t:
        t.extractall(path="data/wiki")
        print(f"Downloaded data/wiki")
    os.remove("data/wiki-rolling_nips.tar.gz")
    os.rename("data/wiki/wiki-rolling_nips/", "data/wiki/wiki_nips")
    os.rename("data/wiki/wiki_nips/train/train.json", "data/wiki/wiki_nips/train/data.json")
    os.rename("data/wiki/wiki_nips/test/test.json", "data/wiki/wiki_nips/test/data.json")

elif sys.argv[1] == "msl":
    # Install MSL dataset 

    url = 'https://drive.google.com/uc?id=14STjpszyi6D0B7BUHZ1L4GLUkhhPXE0G'
    gdown.download(url, 'MSL.zip', quiet=False)
    with zipfile.ZipFile('MSL.zip') as z:
        z.extractall("data/anomaly_detection")
    os.remove('MSL.zip')
    print(f"Downloaded MSL dataset")

elif sys.argv[1] == "psm":
    # Install PSM dataset 

    url = "https://drive.google.com/uc?id=14gCVQRciS2hs2SAjXpqioxE4CUzaYkhb"
    gdown.download(url, 'PSM.zip', quiet=False)
    with zipfile.ZipFile('PSM.zip') as z:
        z.extractall("data/anomaly_detection")
    os.remove('PSM.zip')
    print(f"Downloaded PSM dataset")

elif sys.argv[1] == "smap":
    # Install SMAP dataset 

    url = "https://drive.google.com/uc?id=1kxiTMOouw1p-yJMkb_Q_CGMjakVNtg3X"
    gdown.download(url, 'SMAP.zip', quiet=False)
    with zipfile.ZipFile('SMAP.zip') as z:
        z.extractall("data/anomaly_detection")
    os.remove('SMAP.zip')
    print(f"Downloaded SMAP dataset")


elif sys.argv[1] == "smd":
    # Install SMD dataset 

    url = "https://drive.google.com/uc?id=1BgjQ7_2uqRrZ789Pijtpid5xpLTniywu"
    gdown.download(url, 'SMD.zip', quiet=False)
    with zipfile.ZipFile('SMD.zip') as z:
        z.extractall("data/anomaly_detection")
    os.remove('SMD.zip')
    print(f"Downloaded SMD dataset")


elif sys.argv[1] == "swat":
    # Install SWaT dataset 

    url = "https://drive.google.com/uc?id=1eRKQwJhqmUD4LkWnqNy1cdIz3W_y6EtW"
    gdown.download(url, 'SWaT.zip', quiet=False)
    with zipfile.ZipFile('SWaT.zip') as z:
        z.extractall("data/anomaly_detection")
    os.remove('SWaT.zip')
    print(f"Downloaded SWaT dataset")


