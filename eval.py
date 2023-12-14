import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def evaluate_training(model, train_dataloader, validation_dataloader, num_classes=2, device="cpu"):
    f1_score = MulticlassF1Score(device=device, num_classes=num_classes)
    confusion_matrix = MulticlassConfusionMatrix(num_classes, normalize="true")
    model_name = f"[{model.__class__.__name__}]"
    model.eval()

    # Train evaluation
    with torch.no_grad():
        for images, labels  in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            f1_score.update(outputs, labels)
            confusion_matrix.update(outputs, labels)
    print(f"{model_name} Train f1 score: {f1_score.compute()}")
    cm = ConfusionMatrixDisplay(confusion_matrix.compute().numpy(), display_labels=["Normal", "Pneumonia"])
    cm.plot(values_format=".6f", cmap="inferno")
    cm.ax_.set_title(f"{model_name} Train confusion matrix")

    # Validation evaluation
    f1_score.reset()
    confusion_matrix.reset()
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            f1_score.update(outputs, labels)
            confusion_matrix.update(outputs, labels)
    print(f"{model_name} Validation f1 score: {f1_score.compute()}")
    cm = ConfusionMatrixDisplay(confusion_matrix.compute().numpy(), display_labels=["Normal", "Pneumonia"])
    cm.plot(values_format=".6f", cmap="inferno")
    cm.ax_.set_title(f"{model_name} Validation confusion matrix")

def evaluate_test(model, test_dataloader, num_classes=2, device="cpu"):
    f1_score = MulticlassF1Score(device=device, num_classes=num_classes)
    confusion_matrix = MulticlassConfusionMatrix(num_classes, normalize="true")
    model_name = f"[{model.__class__.__name__}]"
    model.eval()

    # Test evaluation
    with torch.no_grad():
        for batch in test_dataloader:
            if batch.__class__.__name__ == "dict":
                images, labels = batch["image"].to(device),  batch["labels"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            f1_score.update(outputs, labels)
            confusion_matrix.update(outputs, labels)
    print(f"{model_name} Test f1 score: {f1_score.compute()}")
    cm = ConfusionMatrixDisplay(confusion_matrix.compute().numpy(), display_labels=["Normal", "Pneumonia"])
    cm.plot(values_format=".6f", cmap="inferno")
    cm.ax_.set_title(f"{model_name} Test confusion matrix")
    
def plot_loss(model):
    model_name = f"[{model.__class__.__name__}]"
    _, ax = plt.subplots()
    train_x_axis = np.linspace(1, model.history["epochs"], num=len(model.history["train_loss"]))
    val_x_axis = np.linspace(1, model.history["epochs"], num=len(model.history["validation_loss"]))
    ax.plot(train_x_axis, model.history["train_loss"], color="#407cdb", label="Train")
    ax.plot(val_x_axis, model.history["validation_loss"], color="#db5740", label="Validation")

    ax.legend(loc="upper left")
    handles, labels = ax.get_legend_handles_labels()
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{model_name} loss vs. epochs")
    plt.show()