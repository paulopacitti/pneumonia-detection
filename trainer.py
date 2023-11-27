# trainer of the deep learning model in model.py
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader,  optimizer, loss_fn, device, epochs, classes=2):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_loss = []
        self.val_loss = []
        self.epochs = epochs
        self.classes = 2

    def fit(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # Zero your gradients for every batch
                self.optimizer.zero_grad()

                # Make predictions for this batch
                Y_pred = self.model.forward(images)

                # Compute the loss and its gradients
                loss_value = self.loss_fn(Y_pred, labels)
                train_loss += loss_value.item()
                loss_value.backward()

                # Adjust learning weights
                self.optimizer.step()
                if i % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.epochs, i, len(self.train_dataloader), loss_value.item()))
            self.train_loss.append(train_loss / len(self.train_dataloader))

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, (images, labels) in enumerate(self.val_dataloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    Y_pred = self.model.forward(images)
                    loss_value = self.loss_fn(Y_pred, labels)
                    val_loss += loss_value.item()
                print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, self.epochs, val_loss / len(self.val_dataloader)))
                self.val_loss.append(val_loss / len(self.val_dataloader))

    def evaluate(self):
        accuracy = MulticlassAccuracy(device=self.device, num_classes=self.classes)
        f1_score = MulticlassF1Score(device=self.device, num_classes=self.classes)
        confusion_matrix = MulticlassConfusionMatrix(num_classes=self.classes, normalize="true")
        self.model.eval()

        # Train evaluation
        with torch.no_grad():
            for images, labels  in self.train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.forward(images)
                accuracy.update(outputs, labels)
                f1_score.update(outputs, labels)
                confusion_matrix.update(outputs, labels)
        print(f"Train accuracy: {accuracy.compute()}")
        print(f"Train f1 score: {f1_score.compute()}")
        cm = ConfusionMatrixDisplay(confusion_matrix.compute().numpy(), display_labels=['Normal', 'Pneumonia'])
        cm.plot(values_format=".6f", cmap="inferno")
        cm.ax_.set_title('Train confusion matrix')
        
        # Validation evaultation
        accuracy.reset()
        f1_score.reset()
        confusion_matrix.reset()
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.forward(images)
                accuracy.update(outputs, labels)
                f1_score.update(outputs, labels)
                confusion_matrix.update(outputs, labels)
        print(f"Validation accuracy: {accuracy.compute()}")
        print(f"Validation f1 score: {f1_score.compute()}")
        cm = ConfusionMatrixDisplay(confusion_matrix.compute().numpy(), display_labels=['Normal', 'Pneumonia'])
        cm.plot(values_format=".6f", cmap="inferno")
        cm.ax_.set_title("Validation confusion matrix")

    def get_history(self):
        return { "train_loss": self.train_loss, "validation_loss": self.val_loss, "epochs": self.epochs }
    
    def set_history(self, history):
        self.train_loss = history["train_loss"]
        self.val_loss = history["validation_loss"]
        self.epochs = history["epochs"]

    def plot_loss(self):
        _, ax = plt.subplots()
        train_x_axis = np.linspace(1, self.epochs, num=len(self.train_loss))
        val_x_axis = np.linspace(1, self.epochs, num=len(self.val_loss))
        ax.plot(train_x_axis, self.train_loss, color="#407cdb", label="Train")
        ax.plot( val_x_axis, self.val_loss, color="#db5740", label="Validation")

        ax.legend(loc="upper left")
        handles, labels = ax.get_legend_handles_labels()
        lgd = dict(zip(labels, handles))
        ax.legend(lgd.values(), lgd.keys())

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss vs. epochs")
        plt.show()
