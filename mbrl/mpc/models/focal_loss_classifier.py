
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split
from ignite.metrics import Accuracy, ConfusionMatrix

from mbrl.mpc.models.base import MLPRegression, MLPCategorical

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        '''
        @param inputs [tensor, (B, C)]: log-softmax output
        @param targets [tensor, (B)]
        '''
        target = target.view(-1,1)

        logpt = inputs.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DataBuffer:
    # numpy-based ring buffer to store data

    def __init__(self, input_dim, output_dim, max_len=5000000):
        self.input_buf = np.zeros(combined_shape(max_len, input_dim), dtype=np.float32)
        self.output_buf = np.zeros(combined_shape(max_len, output_dim), dtype=np.float32)
        self.ptr, self.max_size = 0, max_len
        self.full = 0 # indicate if the buffer is full and begin a new circle
        self.ptr_old = 0
        self.ptr_reset = 0 # indicate if we have reset prt to 0 since last data retrival

    def store(self, input_data, output_data):
        """
        Append one data to the buffer.
        @param input_data [ndarray, input_dim]
        @param input_data [ndarray, output_dim]
        """
        if self.ptr == self.max_size:
            self.full = 1 # finish a ring
            self.ptr_reset += 1
            self.ptr = 0
        self.input_buf[self.ptr] = input_data
        self.output_buf[self.ptr] = output_data
        self.ptr += 1

    def get_all(self):
        '''
        Return all the valid data in the buffer
        @return input_buf [ndarray, (size, input_dim)], output_buf [ndarray, (size, input_dim)]
        '''

        self.ptr_reset = 0
        if self.full:
            print("data buffer is full, return all data: ", self.max_size, self.ptr)
            return self.input_buf, self.output_buf
        # Buffer is not full
        print("return data util ", self.ptr)
        return self.input_buf[:self.ptr], self.output_buf[:self.ptr]

    def get_new(self):
        '''
        Return new input data in the buffer since last retrival by calling this method
        @return input_buf [ndarray, (size, input_dim)], output_buf [ndarray, (size, input_dim)]
        '''
        if self.ptr_reset > 1: # two round
            x, y = self.input_buf, self.output_buf
        elif self.ptr_reset == 1:
            if self.ptr < self.ptr_old:
                x = np.concatenate((self.input_buf[:self.ptr], self.input_buf[self.ptr_old]), axis=1)
                y = np.concatenate((self.output_buf[:self.ptr], self.output_buf[self.ptr_old]), axis=1)
            else:
                x, y = self.input_buf, self.output_buf
        else:
            x, y = self.input_buf[self.ptr_old:self.ptr], self.output_buf[self.ptr_old:self.ptr]
        self.ptr_old = self.ptr
        self.ptr_reset = 0
        return x, y


class Classifier:
    def __init__(self, input_dim, output_dim, config):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.save = config["save"]
        self.save_freq = config["save_freq"]
        self.save_path = config["save_path"]
        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        self.gamma=config["gamma"]
        self.alpha=config["alpha"]
        self.dropout = config["dropout"]
        activ = config["activation"]
        activ_f = nn.Tanh if activ.lower() == "tanh" else nn.ReLU

        self.mu = torch.tensor(0.0)
        self.sigma = torch.tensor(1.0)
        self.eps = 0.001

        if config["load"]:
            self.load_model(config["load_path"])
            print("successfually load model from ", config["load_path"])
        else:
            self.model = CUDA(MLPCategorical(self.input_dim, self.output_dim, config["hidden_sizes"], activation=activ_f, dropout=self.dropout))

        self.criterion = FocalLoss(gamma=self.gamma, alpha=self.alpha, size_average=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.acc_train, self.acc_test = Accuracy(), Accuracy()
        self.conf_train, self.conf_test = ConfusionMatrix(num_classes=2), ConfusionMatrix(num_classes=2)

        self.data_buf = DataBuffer(self.input_dim, self.output_dim, max_len=config["buffer_size"])
    
    def add_data_point(self, input_data, output_data):
        '''
        This method is used for streaming data setting, where one data will be added at each time.
        @param input_data [list or ndarray, (input_dim)]
        @param output_data [list or ndarray, (output_dim)]
        '''
        x = np.array(input_data).reshape(self.input_dim)
        y = np.array(output_data).reshape(self.output_dim)
        self.data_buf.store(x, y)
        
    def predict(self, data):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param data [list or ndarray, (batch, input_dim)]
        @return out [list or ndarray, (batch, output_dim)]
        '''
        self.model.eval()
        inputs = data if torch.is_tensor(data) else torch.tensor(data).float()
        inputs = (inputs-self.mu) / (self.sigma)
        inputs = CUDA(inputs)
        with torch.no_grad():
            out = self.model(inputs)
            out = CPU(out.argmax(dim=1, keepdim=True))  # get the index, [batch, 1])
            out = out.numpy()
        return out
            
    def make_dataloader(self, x, y, normalize = True):
        '''
        This method is used to generate dataloader object for training.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''
        
        tensor_x = x if torch.is_tensor(x) else torch.tensor(x).float()
        tensor_y = y if torch.is_tensor(y) else torch.tensor(y).long()

        num_data = tensor_x.shape[0]

        if normalize:
            self.mu = torch.mean(tensor_x, dim=0, keepdims=True)
            self.sigma = torch.std(tensor_x, dim=0, keepdims=True)
            self.sigma[self.sigma<self.eps] = 1
            print("data normalized")
            print("mu: ", self.mu)
            print("sigma: ", self.sigma)
            tensor_x = (tensor_x-self.mu) / (self.sigma)

        dataset = TensorDataset(tensor_x, tensor_y)

        testing_len = int(self.test_ratio * num_data)
        training_len = num_data - testing_len
        train_set, test_set = random_split(dataset, [training_len, testing_len])
        train_loader = DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        return train_loader, test_loader

    def fit(self, x=None, y=None, use_data_buf=True, normalize=True):
        '''
        Train the model either from external data or internal data buf.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch, output_dim)]
        '''
        if use_data_buf:
            x, y = self.data_buf.get_all()
            train_loader, test_loader =  self.make_dataloader(x, y, normalize = normalize)
        else: # use external data loader
            train_loader, test_loader = self.make_dataloader(x, y, normalize = normalize)
        
        for epoch in range(self.n_epochs):
            loss_train= 0
            self.model.train()
            self.acc_train.reset()
            self.conf_train.reset()

            for datas, labels in train_loader:
                datas = CUDA(datas)
                labels = CUDA(labels)

                self.optimizer.zero_grad()
                outputs = self.model(datas)
                labels = torch.squeeze(labels)
                self.acc_train.update((outputs, labels))
                self.conf_train.update((outputs, labels))

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()*datas.shape[0] # sum of the loss
                #pred = outputs.argmax(dim=1, keepdim=True)  # get the index
                #acc_train += pred.eq(labels.view_as(pred)).sum().item()

            if self.save and (epoch+1) % self.save_freq == 0:
                self.save_model(self.save_path)
                
            if (epoch+1) % self.test_freq == 0:
                loss_train /= len(train_loader.dataset)

                acc_train = self.acc_train.compute()
                tn, fp, fn, tp = self.conf_train.compute().numpy().flatten()
                fpr = fp / (fp + tn) # false positive rate
                fnr = fn / (fn + tp) # false negative rate
    
                if len(test_loader) > 0:
                    loss_test, acc_test, fpr_test,fnr_test = self.test_model(test_loader)
                    print(f"[{epoch}/{self.n_epochs}],train l|acc|FPR|FNR: {loss_train:.4f}|{100.*acc_train:.2f}%|{100.*fpr:.2f}|{100.*fnr:.2f}%, test: {loss_test:.4f}|{100.*acc_test:.2f}%|{100.*fpr_test:.2f}|{100.*fnr_test:.2f}%")
                else:
                    print(f"epoch[{epoch}/{self.n_epochs}],train l|acc: {loss_train:.4f}|{100.*acc_train:.2f}%")
        
        if self.save:
            self.save_model(self.save_path)

    def test_model(self, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        self.model.eval()
        loss_test = 0
        self.acc_test.reset()
        self.conf_test.reset()
        for datas, labels in testloader:
            datas = CUDA(datas)
            labels = CUDA(labels)
            outputs = self.model(datas)
            labels = torch.squeeze(labels)
            self.acc_test.update((outputs, labels))
            self.conf_test.update((outputs, labels))
            loss = self.criterion(outputs, labels)
            loss_test += loss.item()*datas.shape[0]
            #pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #acc_test += pred.eq(labels.view_as(pred)).sum().item()
        loss_test /= len(testloader.dataset)
        #acc_test /= len(testloader.dataset)
        acc_test = self.acc_test.compute()
        tn, fp, fn, tp = self.conf_test.compute().numpy().flatten()
        fpr = fp / (fp + tn) # false positive rate
        fnr = fn / (fn + tp) # false negative rate
        self.model.train()
        return loss_test, acc_test, fpr, fnr

    def transform(self, x):
        '''
        @param x - [ndarray or tensor, (batch, input_dim)]
        '''
        self.model.eval()
        inputs = x if torch.is_tensor(x) else torch.tensor(x).float()
        inputs = (inputs-self.mu) / (self.sigma)
        inputs = CUDA(inputs)
        with torch.no_grad():
            out = self.model(inputs)
            out = CPU(out.data.exp())  # get the softmax representation, [batch, 2])
            out = out.numpy()
        return out

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        
        checkpoint = {"model_state_dict":self.model, "mu":self.mu, "sigma":self.sigma}
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model_state_dict"]
        self.model = CUDA(self.model)
        self.mu = checkpoint["mu"]
        self.sigma = checkpoint["sigma"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)