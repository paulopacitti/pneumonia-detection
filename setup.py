# setup environment
import os
import zipfile
import gdown
from torch.utils.data import DataLoader 
import datasets

def setup_train_data(data_folder="data"):
    print("[Setup dataset]:")
    file_id = "1qVY01kUnNdX92N6vb1PtzAy9SzxJDcR_"

    if not os.path.isdir("data_1"):
        os.mkdir(f"{data_folder}")

        gdown.download(id=file_id, output=f"{data_folder}/chest_xray.zip", quiet=False)
        with zipfile.ZipFile(f"{data_folder}/chest_xray.zip", "r") as zip_ref:
            zip_ref.extractall(f"{data_folder}")
        
        os.remove(f"{data_folder}/chest_xray.zip")

def setup_nih_test_data(data_folder="data"):
    class_normal_idx, class_pneumonia_idx = [0, 7] # 0 stands for NORMAL and 7 stands for PNEUMONIA
    ds = datasets.load_dataset(
        "alkzar90/NIH-Chest-X-ray-dataset", 
        "image-classification", 
        split="test", 
    )
    ds = ds.filter(lambda x: class_normal_idx in x["labels"] or class_pneumonia_idx in x["labels"])
    ds = ds.map(relabel_nih_test_dataset)
    ds.with_format("torch")
    ds.save_to_disk(f"{data_folder}/nih_test_dataset")

def relabel_nih_test_dataset(e):
    class_normal_idx, class_pneumonia_idx = [0, 7] # 0 stands for NORMAL and 7 stands for PNEUMONIA
    if class_normal_idx in e["labels"]:
        e["labels"] = 0
    elif class_pneumonia_idx in e["labels"]:
        e["labels"] = 1
    return e

if __name__ == "__main__":
    setup_train_data()
    setup_nih_test_data()
    