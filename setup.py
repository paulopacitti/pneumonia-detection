# setup environment
import os
import zipfile
import gdown

def setup_data(data_folder="data"):
    print("[Setup dataset]:")
    file_id = "1qVY01kUnNdX92N6vb1PtzAy9SzxJDcR_"

    if not os.path.isdir("data_1"):
        os.mkdir(f"{data_folder}")

        gdown.download(id=file_id, output=f"{data_folder}/chest_xray.zip", quiet=False)
        with zipfile.ZipFile(f"{data_folder}/chest_xray.zip", "r") as zip_ref:
            zip_ref.extractall(f"{data_folder}")
        
        os.remove(f"{data_folder}/chest_xray.zip")

if __name__ == "__main__":
    setup_data()