'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-05-24 14:46:10
@LastEditTime: 2020-06-21 13:17:21
@Description:
'''
from mbrl.mpc.models.base import GRURegression
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

def default_config():
    config = dict(
        n_epochs=200,
        learning_rate=0.001,
        batch_size=64,
        embedding_sizes=(256, 256),
        output_sizes=(100, 100),

        save=False,
        save_freq=100,
        save_path="default_model.ckpt",
        test_freq=10,
        test_ratio=0.2,
        activation="elu",
        load=False,
    )
    return config



class RSSM:
    def __init__(self, input_dim, output_dim, config=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if config is None:
            config = default_config()

        self.n_epochs = config["n_epochs"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.embedding_sizes = config["embedding_sizes"]
        self.output_sizes = config["output_sizes"]

        self.save = config["save"]
        self.save_freq = config["save_freq"]
        self.save_path = config["save_path"]
        self.test_freq = config["test_freq"]
        self.test_ratio = config["test_ratio"]
        activ = config["activation"]
        if activ == "relu":
            activ_f = nn.ReLU
        elif activ == "tanh":
            activ_f = nn.Tanh
        elif activ == "elu":
            activ_f = nn.ELU
        else:
            activ_f = nn.ReLU

        if config["load"]:
            self.load_model(config["load_path"])
        else:
            self.model = CUDA(GRURegression(self.input_dim, self.output_dim, self.embedding_sizes, self.output_sizes, activation=activ_f))

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def rollout_state(self, steps, prev_states, actions, init_hidden=None, pred_delta = False ):
        """
        rollout to predict future states based on previous states and given actions
        ----------
            @param steps [int] : number of steps to rollout
            @param prev_states [tensor, (prev T, batch, state dim)] : history states
            @param actions [tensor, (prev T+steps+1, batch, action dim)] : history actions and future actions want to rollout
            @param init_hidden [tensor, (batch, hidden dim)] : hidden states from the initial timestep
        ----------
            @return rollouts [tensor, (steps, batch, state dim)]
            @return hidden [tensor, (batch, hidden dim)] : hidden state output of the last timestep
        """
        prev_length, batch_size, state_dim = prev_states.shape
        assert prev_length+steps == actions.shape[0]+1, "Timesteps for arguments 'actions', 'prev_states', and 'steps' do not match "

        rollouts = CUDA(torch.zeros(steps, batch_size, state_dim))

        prev_states = CUDA(prev_states)
        actions = CUDA(actions)
        prev_actions = actions[:prev_length] # [T, B, a]
        prev_inputs = torch.cat((prev_states, prev_actions), dim=2) # [T, B, s+a]
        if init_hidden is None:
            hidden = self.model.initial_hidden(batch_size)
        else:
            hidden = CUDA(init_hidden)

        if pred_delta:
            # rollout the history states to get the hidden states
            for t in range(prev_length):
                delta_states, hidden = self.model(prev_inputs[t], hidden)
                states = delta_states + prev_states[t]

            rollouts[0] = states
            # rollout the future states
            for t in range(steps-1):
                inputs = torch.cat((states, actions[prev_length+t]), dim=1) # [B, s+a]
                delta_states, hidden = self.model(inputs, hidden)
                states = delta_states + states
                rollouts[t+1] = states
        else:
            # rollout the history states to get the hidden states
            for t in range(prev_length):
                states, hidden = self.model(prev_inputs[t], hidden)

            rollouts[0] = states
            # rollout the future states
            for t in range(steps-1):
                inputs = torch.cat((states, actions[prev_length+t]), dim=1) # [B, s+a]
                states, hidden = self.model(inputs, hidden)
                rollouts[t+1] = states

        return rollouts, hidden

    def loss(self, states, actions, labels, history_len = 1, length = None, debug=False, pred_delta = False):
        '''
        Calculate the loss based on history trajectories
        ----------
            @param states [tensor, (T, batch, state dim)] : padded history states sequence
            @param actions [tensor, (T, batch, action dim)] : padded history actions sequence
            @param labels [tensor, (T, batch, state dim)] : padded history states_next
            @param history_len [int] : use history-len data as the input to rollout other steps
            @param length [tensor, (batch)] : length of each unpadded trajectory in the batch
        ----------
            @return loss [tensor, (1)]
        '''
        
        T, batch_size, state_dim = states.shape
        assert history_len<T, "no more future trajectories to rollout!"
        steps = T - history_len + 1
        prev_states = states[:history_len] # [h_len, B, s]

        if debug:
            print("steps: ", steps, " T: ", T, " history_len: ", history_len)
        rollouts, _ = self.rollout_state(steps, prev_states, actions,pred_delta=pred_delta) # [steps, B, s]
        if debug:
            print("rollouts shape: ", rollouts.shape)

        if pred_delta:
            targets = labels[-steps:] - states[-steps:]
        else:
            targets = labels[-steps:] # [steps, B, s]
        if length is None:
            loss = self.criterion(rollouts, targets)
        else:
            #print("lengths: ", length)
            length = length - history_len + 1
            length[length<0] = 0
            if debug:
                print("lengths: ", length)
            packed_rollouts = pack_padded_sequence(rollouts, length, enforce_sorted=True).data
            packed_label = pack_padded_sequence(targets, length, enforce_sorted=True).data
            if debug:
                print("packed rollouts shape: ", packed_rollouts)
                print("packed label shape: ", packed_label)
            loss = self.criterion(packed_rollouts, packed_label)
        return loss

    def fit(self, train_loader, test_loader=None, history_len = 1):
        '''
        Shape of data: [T, B, dim]
        '''
        data_num = len(train_loader.dataset)

        for epoch in range(self.n_epochs):
            self.model.train()
            loss_train = 0
            for state, action, label, length in train_loader:
                state = CUDA(state)
                action = CUDA(action)
                label = CUDA(label)
                self.optimizer.zero_grad()
                loss = self.loss(state, action, label, history_len=1, length=length)
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()*state.shape[1] # sum of the loss

            if self.save and (epoch+1) % self.save_freq == 0:
                self.save_model(self.save_path)

            if (epoch+1) % self.test_freq == 0:
                loss_train /= data_num
                loss_test = -0.1234
                if test_loader is not None:
                    loss_test = self.test_model(test_loader)
                    print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, loss test  {loss_test:.4f}")
                else:
                    print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, no testing data")
                #loss_unormalized = self.test(x[::50], y[::50])
                #print("loss unnormalized: ", loss_unormalized)
                
    def test_model(self, testloader):
        '''
        Test the model with normalized test dataset.
        @param test_loader [torch.utils.data.DataLoader]
        '''
        self.model.eval()
        loss_test = 0
        for state, action, label, length in testloader:
            state = CUDA(state)
            action = CUDA(action)
            label = CUDA(label)
            self.optimizer.zero_grad()
            loss = self.loss(state, action, label, length=length)
            loss_test += loss.item()*state.shape[1]
        loss_test /= len(testloader.dataset)
        self.model.train()
        return loss_test

    def evaluate_rollout(self, states, actions, labels, history_len = 1, length = None, eval_dim = None, debug=False):
        '''
        Calculate the mse loss along time steps
        ----------
            @param states [tensor, (T, batch, state dim)] : padded history states sequence
            @param actions [tensor, (T, batch, action dim)] : padded history actions sequence
            @param labels [tensor, (T, batch, state dim)] : padded history states_next
            @param history_len [int] : use history-len data as the input to rollout other steps
            @param length [tensor, (batch)] : length of each unpadded trajectory in the batch (must be sorted)
            @param eval_dim [slice or None] : determine which state dims will be evaluated. If None, evaluate all dims.
        ----------
            @return loss [list, (T - his_len + 1)]
        '''
        states = CUDA(states)
        actions = CUDA(actions)
        labels = CUDA(labels)
        
        T, batch_size, state_dim = states.shape
        assert history_len<T, "no more future trajectories to rollout!"
        steps = T - history_len + 1
        prev_states = states[:history_len] # [h_len, B, s]

        if debug:
            print("steps: ", steps, " T: ", T, " history_len: ", history_len)
        rollouts, _ = self.rollout_state(steps, prev_states, actions) # [steps, B, s]
        if debug:
            print("rollouts shape: ", rollouts.shape)
        targets = labels[-steps:] # [steps, B, s]

        if eval_dim is not None:
            targets = targets[:,:, eval_dim] #[steps, B, dim]
            rollouts = rollouts[:,:,eval_dim] #[steps, B, dim]

        MSE = torch.mean( (targets-rollouts)**2, dim=2 ) #[steps, B]
        if length is None:
            loss = torch.mean(MSE, dim=1) # [steps]
            loss = list(CPU(loss).numpy())
        else:
            #print("lengths: ", length)
            length = length - history_len + 1
            if debug:
                print("lengths: ", length)
            loss = []
            for t in range(steps):
                mask = length-t
                mask[mask>0] = 1
                mask[mask<0] = 0
                B = int(torch.sum(mask).item())
                mse = MSE[t][:B] # [B]
                loss.append(torch.mean(mse).item())
        return loss

    def reset_model(self):
        self.model.apply(self.weight_init)

    def add_trajectory(self):
        """
        Preprocess the observation and add to the regression model data buffer
        @param obs, obs_next [ndarray]
        """
        #obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        pass

    def weight_init(m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)