'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-07-10 10:42:50
@LastEditTime: 2020-07-14 17:38:57
@Description:
'''

import numpy as np
import joblib
import os.path as osp
import os
import time

import lightgbm as lgb

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class DataBuffer:
    # numpy-based ring buffer to store data

    def __init__(self, input_dim, output_dim=None, max_len=5000000):
        self.input_buf = np.zeros(combined_shape(max_len, input_dim), dtype=np.float32)
        self.output_buf = np.zeros(combined_shape(max_len), dtype=np.float32)
        self.ptr, self.max_size = 0, max_len
        self.full = 0 # indicate if the buffer is full and begin a new circle

    def store(self, input_data, output_data):
        """
        Append one data to the buffer.
        @param input_data [ndarray, input_dim]
        @param input_data [ndarray, output_dim]
        """
        if self.ptr == self.max_size:
            self.full = 1 # finish a ring
            self.ptr = 0
        self.input_buf[self.ptr] = input_data
        self.output_buf[self.ptr] = output_data
        self.ptr += 1

    def get_all(self):
        '''
        Return all the valid data in the buffer
        @return input_buf [ndarray, (size, input_dim)], output_buf [ndarray, (size, input_dim)]
        '''
        if self.full:
            print("data buffer is full, return all data: ", self.max_size, self.ptr)
            return self.input_buf, self.output_buf
        # Buffer is not full
        print("return data util ", self.ptr)
        return self.input_buf[:self.ptr], self.output_buf[:self.ptr]

    def save(self, path=None):
        assert path is not None, "The saving path is not specified!"
        x, y = self.get_all()
        data = {"x":x,"y":y}
        joblib.dump(data, path)
        print("Successfully save data buffer to "+path)

    def load(self, path=None):
        assert path is not None, "The loading path is not specified!"
        data = joblib.load(path)
        x, y = data["x"], data["y"]
        data_num = x.shape[0]
        if data_num<self.max_size:
            self.input_buf[:data_num], self.output_buf[:data_num] = x, y
            self.ptr = data_num
        else:
            self.input_buf, self.output_buf = x[:self.max_size], y[:self.max_size]
            self.full = 1

default_config = dict(
    model_param=dict(
                boosting_type='gbdt', 
                num_leaves=12, 
                max_depth=8, 
                learning_rate=0.1, 
                n_estimators=800,
                ),
    safe_buffer_size=10000,
    unsafe_buffer_size=50000,
    uncertain_buffer_size=500,
    batch=1000,
    max_ratio=2.5,
    save=True,
    save_folder="../data/cost_model_pg1",
    load=False,
    load_folder="../data/cost_model_pg1",
    )

class CostModel:
    def __init__(self, env_wrap, config=default_config):
        super().__init__()
        self.env_wrap = env_wrap
        self.idx_goal = env_wrap.key_to_slice["goal"]
        self.state_dim = env_wrap.observation_size
        self.action_dim = env_wrap.action_size

        self.uncertain_buffer_size = config["uncertain_buffer_size"]

        self.safe_data_buf = DataBuffer(self.state_dim, max_len=config["safe_buffer_size"])
        self.unsafe_data_buf = DataBuffer(self.state_dim, max_len=config["unsafe_buffer_size"])
        self.uncertain_data_buf = DataBuffer(self.state_dim, max_len=self.uncertain_buffer_size)

        self.unsafe_transition_data_buf = DataBuffer(self.state_dim+self.action_dim, max_len=config["unsafe_buffer_size"])

        self.max_ratio = config["max_ratio"]
        self.batch = int(config["batch"])
        
        self.model = self.load_data(config["load_folder"]) if config["load"] else None
        if self.model is None:
            self.model = lgb.LGBMClassifier(**config["model_param"])
        else:
            self.model.set_params(**config["model_param"])

        self.save = config["save"]
        if self.save:
            self.folder = config["save_folder"]
            if osp.exists(self.folder):
                print("Warning: Saving dir %s already exists! Storing model and buffer there anyway."%self.folder)
            else:
                os.makedirs(self.folder)
            self.safe_data_buf_path = osp.join(self.folder, "safe_data_buf.pkl")
            self.unsafe_data_buf_path = osp.join(self.folder, "unsafe_data_buf.pkl")
            self.uncertain_data_buf_path = osp.join(self.folder, "uncertain_data_buf.pkl")
            self.model_path = osp.join(self.folder, "cost_model.pkl")

    def add_data_point(self, state, cost, s_a_pair):
        '''
        This method is used for streaming data setting, where one data will be added at each time.
        @param state [list or ndarray, (state_dim)]
        @param s_a_pair [list or ndarray, (state_dim + action_dim)]: unsafe state action pair
        @param cost [int, (1)]: 0 - safe data; 1 - unsafe data; -1 - uncertain data
        '''
        x = np.array(state).reshape(self.state_dim)
        if cost == 0:
            self.safe_data_buf.store(x, 0)
        elif cost > 0:
            self.unsafe_data_buf.store(x, 1)
            self.unsafe_transition_data_buf.store(s_a_pair, 1)
        elif cost < 0:
            #self.uncertain_data_buf.store(x, 1)
            print("The cost should not be negative!!!!!!!!!!!!!!!!!!!!")

    def get_uncertain_data(self, dynamic_model): 
        x, y = self.unsafe_transition_data_buf.get_all()
        assert x.shape[0]>0, " no unsafe data buffer "
        state_next = dynamic_model.predict(x) #[B, s]
        constraints_violation = self._predict_cost(state_next) # [B]
        inaccurate_pred = state_next[constraints_violation==0] # If dynamic model is accurate, all the prediction should be 1
        num = inaccurate_pred.shape[0]
        total_num = x.shape[0]
        print("inaccurate_pred: %i/%i, ratio: %.3f"%(num, total_num, num/total_num))
        return inaccurate_pred

    def model_correction(self, dynamic_model, iteration=3):
        print("Conducting safety check with collected unsafe data....")
        for i in range(iteration):
            print("round ", i)
            inaccurate_pred = self.get_uncertain_data(dynamic_model)
            x0, y0 = self.safe_data_buf.get_all()
            x1, y1 = self.unsafe_data_buf.get_all()
            x2, y2 = inaccurate_pred, np.ones(inaccurate_pred.shape[0])
            num0, num1 = x0.shape[0], x1.shape[0]+x2.shape[0]
            max_num0 = int(num1*self.max_ratio+1)  # balance the safe and unsafe data ratio
            num0 = max_num0 if num0>max_num0 else num0
            X = np.concatenate((x0[-num0:], x1, x2), axis=0)
            Y = np.concatenate((y0[:num0], y1, y2), axis=0)
            self.fit(X,Y, use_data_buf=False)

        
    def predict(self, state):
        '''
        This method perform regression with ndarray data and output ndarray data
        @param state [ndarray, (batch, input_dim)]
        @return cost [ndarray, (batch,)]
        '''

        goal_pos = state[:, self.idx_goal] # [batch, 2]
        dist = np.sqrt(np.sum(goal_pos**2, axis=1)) #[batch], cost = x^2+y^2
        dist[dist<0.2] += -5
        dist[dist<0.1] += -8
        cost = dist # Goal reward part

        constraints_violation = self._predict_cost(state)

        cost[constraints_violation==1] += 500
        return cost

    def _predict_cost(self, state):
        data_num = state.shape[0]
        if data_num>self.batch:
            constraints_violation = np.zeros(data_num)
            forward_num = int(np.ceil(data_num/self.batch))
            for i in range(forward_num):
                idx = slice(i*self.batch, (i+1)*self.batch)
                constraints_violation[idx] = self.model.predict(state[idx])
        else:
            constraints_violation = self.model.predict(state)
        return constraints_violation

    def fit(self, x=None, y=None, use_data_buf=True):
        '''
        Train the model either from external data or internal data buf.
        @param x [list or ndarray, (batch, input_dim)]
        @param y [list or ndarray, (batch)]
        '''
        if use_data_buf:
            x0, y0 = self.safe_data_buf.get_all()
            x1, y1 = self.unsafe_data_buf.get_all()
            x2, y2 = self.uncertain_data_buf.get_all()
            num0, num1 = x0.shape[0], x1.shape[0]
            max_num0 = int(num1*self.max_ratio+1)  # balance the safe and unsafe data ratio
            num0 = max_num0 if num0>max_num0 else num0
            X = np.concatenate((x0[-num0:], x1, x2), axis=0)
            Y = np.concatenate((y0[:num0], y1, y2), axis=0)
        else: # use external data loader
            X, Y = x, y

        self.model.fit(X, Y)

        y_pred=self.model.predict(x0)
        acc0=np.equal(y0, y_pred)
        y_pred=self.model.predict(x1)
        acc1=np.equal(y1, y_pred)
        print("unsafe acc: %.2f%%, safe acc: %.2f%%"%(100*acc1.mean(), 100*acc0.mean()) )

        if self.save:
            self.save_data()

    def transform(self, x):
        '''
        @param x - [ndarray, (batch, input_dim)]
        @return out - [ndarray, [batch, 2]]
        '''
        x = x.reshape(-1, self.state_dim)
        out = self.model.predict_proba(x)
        return out

    def reset_uncertain_data_buffer(self):
        self.uncertain_data_buf = DataBuffer(self.state_dim, max_len=self.uncertain_buffer_size)

    def save_data(self):
        joblib.dump(self.model, self.model_path)
        self.safe_data_buf.save(self.safe_data_buf_path)
        self.unsafe_data_buf.save(self.unsafe_data_buf_path)
        self.uncertain_data_buf.save(self.uncertain_data_buf_path)
        print("Successfully save model and data buffer to %s"%self.folder)

    def load_data(self, path):
        model_path = osp.join(path, "cost_model.pkl")
        if osp.exists(model_path):
            model = joblib.load(model_path)
            print("Loading model from %s ."%model_path)
        else:
            model = None
            print("We can not find the model from %s"%model_path)
        unsafe_data_buf_path = osp.join(path, "unsafe_data_buf.pkl")
        if osp.exists(unsafe_data_buf_path):
            print("Loading data buffer from %s ."%unsafe_data_buf_path)
            self.unsafe_data_buf.load(unsafe_data_buf_path)
        safe_data_buf_path = osp.join(path, "safe_data_buf.pkl")
        if osp.exists(safe_data_buf_path):
            print("Loading data buffer from %s ."%safe_data_buf_path)
            self.safe_data_buf.load(safe_data_buf_path)
        uncertain_data_buf_path = osp.join(path, "uncertain_data_buf.pkl")
        if osp.exists(uncertain_data_buf_path):
            print("Loading data buffer from %s ."%uncertain_data_buf_path)
            self.uncertain_data_buf.load(uncertain_data_buf_path)
        return model