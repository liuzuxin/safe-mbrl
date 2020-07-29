'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:42:50
@LastEditTime: 2020-07-14 23:12:05
@Description:
'''

import numpy as np
from mbrl.mpc.optimizers import RandomOptimizer, CEMOptimizer


class SafeMPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, env, mpc_config, cost_model = None):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(env.action_space.low) # array (dim,)
        self.action_high = np.array(env.action_space.high) # array (dim,)
        self.action_dim = env.action_space.shape[0]
        self.popsize = conf["popsize"]

        self.particle = conf["particle"]

        self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim])
            self.action_high = np.tile(self.action_high, [self.action_dim ])

       
        lb = np.tile(self.action_low, [self.horizon])
        ub = np.tile(self.action_high, [self.horizon])

        #print("low: ", self.action_low ,self.action_high )
        
        self.optimizer = SafeMPC.optimizers[self.type](sol_dim=self.horizon*self.action_dim,
            popsize=self.popsize,upper_bound=ub,lower_bound=lb,
            max_iters=conf["max_iters"],num_elites=conf["num_elites"],
            epsilon=conf["epsilon"],alpha=conf["alpha"])

        if cost_model is not None:
            self.cost_model = cost_model
            self.optimizer.setup(self.cost_function)
        else:
            print("cost function is None, this should not happen")
            self.optimizer.setup(None) # default cost function
        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, model, state):
        '''
        :param state: model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.model = model
        self.state = state

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        if self.type == "CEM":
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        else:
            pass
        #print(soln)
        action = soln[:self.action_dim]
        return action

    def cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1)) # 
        #print(actions.shape)

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0) # [pop size*particle, state dim]
        #print(state.shape)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            x = np.concatenate((state, action), axis=1)
            #print("horizon: ", t)

            state_next = self.model.predict(x) #+ state

            cost = self.cost_model.predict(state_next)  # compute cost

            cost = cost.reshape(costs.shape)

            costs += cost * self.gamma**t
            state = state_next

        # average between particles
        #costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        #if np.min(costs)>=200:
        #    print("all sampling traj will violate constraints")
        return costs