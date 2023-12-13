# trainer of the deep learning model in model.py
import torch

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
                Y_pred = self.model(images)

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
                    Y_pred = self.model(images)
                    loss_value = self.loss_fn(Y_pred, labels)
                    val_loss += loss_value.item()
                print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, self.epochs, val_loss / len(self.val_dataloader)))
                self.val_loss.append(val_loss / len(self.val_dataloader))

    def get_history(self):
        return { "train_loss": self.train_loss, "validation_loss": self.val_loss, "epochs": self.epochs }