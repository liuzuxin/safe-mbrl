'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-05-24 14:46:10
@LastEditTime: 2020-06-23 18:19:00
@Description:
'''
from mbrl.mpc.models.model import RegressionModel, Classifier
import numpy as np
import torch
import torch.nn as nn

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

class DynamicModel:
    def __init__(self, env_wrap, config):
        super().__init__()
        self.coordinates_dim = 2
        self.base_state_dim = env_wrap.base_state_dim
        self.action_dim = env_wrap.action_dim
        self.idx = env_wrap.key_to_slice
        self.input_dim = self.base_state_dim + self.action_dim + self.coordinates_dim
        self.output_dim = self.base_state_dim + self.coordinates_dim
        self.model = RegressionModel(self.input_dim, self.output_dim, config=config)
        self.base_state_name = env_wrap.base_state_name
        self.constraints_name = env_wrap.constraints_name

        for k in self.base_state_name:
            print("base state keys: ", k)

    def add_data_point(self, action, obs, obs_next):
        """
        Preprocess the observation and add to the regression model data buffer
        @param obs, obs_next [ndarray]
        """
        #obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        delta = obs_next #- obs
        x = np.zeros(self.input_dim)
        y = np.zeros(self.output_dim)
        # fill the first base_state_dim in the x and y
        x[:self.base_state_dim] = obs[self.idx["base_state"]]
        y[:self.base_state_dim] = delta[self.idx["base_state"]]
        # append action to x
        x[self.base_state_dim:self.base_state_dim+self.action_dim] = action
        # add goal coordinates
        x[-self.coordinates_dim:] = obs[self.idx["goal"]]
        y[-self.coordinates_dim:] = delta[self.idx["goal"]]
        self.model.add_data_point(x,y)

        # add hazards coordinates
        pos, pos_delta = obs[self.idx["hazards"]], delta[self.idx["hazards"]]
        num = int(pos.shape[0]/self.coordinates_dim)
        for i in range(num):
            x[-self.coordinates_dim:] = pos[2*i:2*i+2]
            y[-self.coordinates_dim:] = pos_delta[2*i:2*i+2]
            self.model.add_data_point(x,y)


    def predict(self, obs, action):
        """
        Predict a batch of state and action pairs and return numpy array

        Parameters:
        ----------
            @param tensor or numpy - state : size should be (batch_size x state dim)
            @param tensor or numpy - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - state_next - size should be (batch_size x state dim)
        """
        inputs = np.zeros((obs.shape[0], self.input_dim))
        outputs = np.zeros(obs.shape)
        # fill the first base_state_dim
        inputs[:,:self.base_state_dim] = obs[:, self.idx["base_state"]]
        # append action to inputs
        inputs[:, self.base_state_dim:self.base_state_dim+self.action_dim] = action
        # predict goal coordinates
        inputs[:, -self.coordinates_dim:] = obs[:, self.idx["goal"]]

        outputs[:, :self.output_dim] = self.model.predict(inputs)

        # predict constraints coordinates
        offset = self.output_dim
        for name in self.constraints_name:
            pos = obs[:,self.idx[name]]
            num = int(pos.shape[1]/self.coordinates_dim)
            for i in range(num):
                inputs[:, -self.coordinates_dim:] = pos[:, 2*i:2*i+2]
                outputs[:, offset:offset+2] = self.model.predict(inputs)[:,-2:]
                offset += 2

        return outputs

    def fit(self, use_data_buf=True, normalize=True):
        self.model.fit(use_data_buf=use_data_buf, normalize=normalize)

    def reset_model(self):
        self.model.reset_model()