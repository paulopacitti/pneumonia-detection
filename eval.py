import torch
from torcheval.metrics import MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import functional as F
from captum.attr import IntegratedGradients, Occlusion, visualization as viz
import numpy as np
import matplotlib.pyplot as plt

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

def get_wrong_classifications(model, dataloader, device="cpu"):
    wrong_classifications = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_labels = torch.argmax(outputs, dim=1)
            incorrect_indices = (predicted_labels != labels).nonzero().squeeze(dim=1)
            for index in incorrect_indices:
                wrong_classifications.append((images[index], labels[index], predicted_labels[index]))

    return wrong_classifications

def get_true_classification_samples(model, dataloader, samples=32, device="cpu"):
    true_classifications = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_indices = (predicted_labels == labels).nonzero().squeeze(dim=1)
            for index in correct_indices:
                true_classifications.append((images[index], labels[index], predicted_labels[index]))
            
            if len(true_classifications) >= samples:
                break

    return true_classifications[:samples]


def interpret_integrated_gradients(model, image, label):
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(image, target=label, n_steps=200)
    return attributions_ig

def interpret_occlusion(model, image, label):
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(image,
                                       strides=(3, 8, 8),
                                       target=label,
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)
    return attributions_occ

def interpret_viz(model, image, label, classes):
    model.eval()
    model.to("cpu")
    outputs = model(image)
    prediction, _ = torch.max(F.softmax(outputs, 1), 1)
    prediction = prediction.detach().squeeze().item()
    predicted = torch.argmax(outputs, dim=1)
    integrated_gradients = interpret_integrated_gradients(model, image, label)
    occlusion = interpret_occlusion(model, image, label)
    print(f"Label: {classes[label]}")
    print(f"Prediction: {prediction}")
    print(f"Predicted: {classes[predicted]}")

    _ = viz.visualize_image_attr_multiple(np.transpose(integrated_gradients.squeeze().cpu().detach().numpy(), (1,2,0)),
                            np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                            methods=["original_image","heat_map"],
                            show_colorbar=True,
                            signs=["all", "positive"],
                            outlier_perc=1,
                            titles=["Original Image", "Integrated Gradients"])
    
    _ = viz.visualize_image_attr_multiple(np.transpose(occlusion.squeeze().cpu().detach().numpy(), (1,2,0)),
                            np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                            methods=["original_image","heat_map"],
                            show_colorbar=True,
                            signs=["all", "positive"],
                            outlier_perc=1,
                            titles=["Original Image", "Occlusion"])

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