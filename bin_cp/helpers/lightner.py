import torch
import torch.optim as optim
from tqdm import tqdm


class Output(object):
    def __init__(self, y_true, logits, y_pred):
        self.y_true = y_true
        self.logits = logits
        self.y_pred = y_pred

class ModelManager(object):
    """ Works similarly to torch lightning
    Handles model training and all other things unless here we also have smooth training and smooth prediction as well.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None
        self.set_optimizer()
    
    def set_optimizer(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5, weight_decay=5e-4)
    
    def fit(self, train_loader, val_loader, n_epochs=500, patience=10, smoothing_function=None):
        if smoothing_function is None:
            smoothing_function = lambda inputs: inputs
        best_model = None
        best_epoch = None
        best_loss = None
        training_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            # training
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                s_inputs = smoothing_function(inputs)

                outputs = self.model(s_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)        

            # test
            self.model.eval()
            test_loss = 0.0
            ys = []
            y_true = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y_true.append(labels)

                    s_inputs = smoothing_function(inputs)

                    outputs = self.model(s_inputs)
                    loss = self.criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    ys.append(outputs.argmax(dim=1))
                test_loss /= len(val_loader)
            ys = torch.concat(ys)
            y_true = torch.concat(y_true)
            accuracy = (ys == y_true).sum() / y_true.shape[0]
            if best_loss is None or best_loss > test_loss:
                best_loss = test_loss
                best_epoch = epoch
                best_model = self.model.state_dict()
            if epoch - best_epoch > patience:
                print(f"early stopping at epoch {epoch}")
                break
            if epoch % 2 == 0:
                print(f"epoch {epoch}, training loss = {train_loss}, test_loss = {test_loss}, accuracy = {accuracy}")

            training_losses.append(train_loss)
            val_losses.append(test_loss)

        self.model.load_state_dict(best_model)
        return training_losses, val_losses

    def smooth_predict(self, test_loader, n_samples=100, smoothing_function=None):
        if smoothing_function is None:
            smoothing_function = lambda inputs: inputs
        
        self.model.eval()

        y_true = []
        y_pred = []
        logits = []
        for inputs, labels in tqdm(test_loader):
            torch.cuda.empty_cache()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                batch_outputs = []
                for iter in range(n_samples):
                    s_inputs = smoothing_function(inputs)
                    outputs = self.model(s_inputs)
                    batch_outputs.append(outputs)
                batch_outputs = torch.stack(batch_outputs).permute(1, 0, 2)
                logits.append(batch_outputs.cpu())
                y_true.append(labels)
                _, max_class = batch_outputs.max(dim=2)
                maj_bin, _ = max_class.mode()
                y_pred.append(maj_bin)
        y_pred = torch.concat(y_pred)
        y_true = torch.concat(y_true)
        logits = torch.concat(logits)

        return Output(y_pred=y_pred, logits=logits, y_true=y_true)
    
    def smooth_adv_predict(self, test_loader, n_samples=100, smoothing_function=None, adv_class=None, r=0.0, adv_conf=dict()):
        if adv_class is None:
            return self.smooth_predict(test_loader, n_samples=n_samples, smoothing_function=smoothing_function)
        
        if smoothing_function is None:
            smoothing_function = lambda inputs: inputs
        
        self.model.eval()
        adv = adv_class(model=self.model, eps=r, **adv_conf)

        y_true = []
        y_pred = []
        logits = []
        for inputs, labels in tqdm(test_loader):
            torch.cuda.empty_cache()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            adv_inputs = adv(inputs, labels)

            with torch.no_grad():
                batch_outputs = []
                for iter in range(n_samples):
                    s_inputs = smoothing_function(adv_inputs)
                    outputs = self.model(s_inputs)
                    batch_outputs.append(outputs)
                batch_outputs = torch.stack(batch_outputs).permute(1, 0, 2)
                logits.append(batch_outputs.cpu())
                y_true.append(labels)
                _, max_class = batch_outputs.max(dim=2)
                maj_bin, _ = max_class.mode()
                y_pred.append(maj_bin)
        y_pred = torch.concat(y_pred)
        y_true = torch.concat(y_true)
        logits = torch.concat(logits)

        return Output(y_pred=y_pred, logits=logits, y_true=y_true)

    def predict(self, test_loader):
        output = self.smooth_predict(test_loader, n_samples=1)
        return Output(y_pred=output.y_pred, logits=output.logits.squeeze(), y_true=output.y_true)
    
    def predict_batch(self, image_batch):
        torch.cuda.empty_cache()
        self.model.eval()
        inputs = image_batch.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
        y_pred = logits.argmax(dim=1)
        return y_pred, logits