'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-05-29 19:49:11
@LastEditTime: 2020-07-15 11:22:04
@Description:
'''
import numpy as np
import re
import gym
import safety_gym

ROBOTS = ['Point','Car', 'Doggo']
TASKS = ['Goal', 'Button']

XYZ_SENSORS = dict(
    Point=['velocimeter'],
    Car=['velocimeter'],#,'accelerometer'],#,'ballquat_rear', 'right_wheel_vel', 'left_wheel_vel'],
    Doggo=['velocimeter','accelerometer']
    )

ANGLE_SENSORS = dict(
    Point=['gyro','magnetometer'],
    Car=['magnetometer','gyro'],
    Doggo=['magnetometer','gyro']
    )

CONSTRAINTS = dict(
    Goal=['vases', 'hazards'],
    Button=['hazards','gremlins','buttons'],)

DEFAULT_CONFIG = dict(
    action_repeat=5,
    max_episode_length=1000,
    use_dist_reward=False,
    stack_obs=False,
)

class Dict2Obj(object):
    #Turns a dictionary into a class
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])
        
    def __repr__(self):
        return "%s" % self.__dict__

class SafetyGymEnv():
    def __init__(self, robot='Point', task='Goal', level=1, seed=0, config=DEFAULT_CONFIG):
        self.robot = robot.capitalize()
        self.task = task.capitalize()
        assert self.robot in ROBOTS, "can not recognize the robot type {}".format(robot)
        assert self.task in TASKS, "can not recognize the task type {}".format(task)
        self.config = Dict2Obj(config)
        env_name = 'Safexp-'+self.robot+self.task+str(level)+'-v0'
        print("Creating environment: ", env_name)
        self.env = gym.make(env_name)
        self.env.seed(seed)

        print("Environment configuration: ", self.config)
        self.init_sensor()

         #for uses with ppo in baseline
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (self.env.action_space.shape[0],), dtype=np.float32)

    def init_sensor(self):
        self.xyz_sensors = XYZ_SENSORS[self.robot]
        self.angle_sensors = ANGLE_SENSORS[self.robot]
        self.constraints_name = CONSTRAINTS[self.task]
        #self.distance_name = ["goal_dist"] + [x+"_dist" for x in self.constraints_name]

        self.base_state_name = self.xyz_sensors + self.angle_sensors
        self.flatten_order = self.base_state_name + ["goal"] + self.constraints_name #+ self.distance_name

        # get state space vector size
        self.env.reset()
        obs = self.get_obs()
        #print(obs)
        self.obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        self.state_dim = self.obs_flat_size
        if self.config.stack_obs:
            self.state_dim = self.state_dim*self.config.action_repeat
        self.key_to_slice = {}
        offset = 0
        for k in self.flatten_order:
            k_size = np.prod(obs[k].shape)
            self.key_to_slice[k] = slice(offset, offset + k_size)
            print("obs key: ", k, " slice: ", self.key_to_slice[k])
            offset += k_size

        self.base_state_dim = sum([np.prod(obs[k].shape) for k in self.base_state_name])
        self.action_dim = self.env.action_space.shape[0]
        self.key_to_slice["base_state"] = slice(0, self.base_state_dim)

    def reset(self):
        self.t = 0    # Reset internal timer
        self.env.reset()
        obs = self.get_obs_flatten()
        
        if self.config.stack_obs:
            for k in range(self.config.action_repeat):
                cat_obs = obs if k == 0 else np.concatenate((cat_obs, obs))
            return cat_obs
        else:
            return obs
    
    def step(self, action):
        # 2 dimensional numpy array, [vx, w]
        
        reward = 0
        cost = 0
        '''
        -------------------------------
        Add low level PID control here
        May need to calculate target velocities for robots actuators first.
        -------------------------------
        '''
        targets = self.calculate_target_control(action)
        if self.config.stack_obs:
            cat_obs = np.zeros(self.config.action_repeat*self.obs_flat_size)

        for k in range(self.config.action_repeat):
            control = action #it should be self.PID_controller(targets, obs)
            state, reward_k, done, info = self.env.step(control)
            if self.config.use_dist_reward:
                reward_k = self.get_dist_reward()
            reward += reward_k
            cost += info["cost"]
            self.t += 1    # Increment internal timer
            observation = self.get_obs_flatten()
            if self.config.stack_obs:
                cat_obs[k*self.obs_flat_size :(k+1)*self.obs_flat_size] = observation 
            goal_met = ("goal_met" in info.keys()) # reach the goal
            done = done or self.t == self.config.max_episode_length
            if done or goal_met:
                if k != self.config.action_repeat-1 and self.config.stack_obs:
                    for j in range(k+1,self.config.action_repeat):
                        cat_obs[j*self.obs_flat_size :(j+1)*self.obs_flat_size] = observation 
                break
        cost = 1 if cost>0 else 0

        info = {"cost":cost, "goal_met":goal_met}
        if self.config.stack_obs:
            return cat_obs, reward, done, info
        else:
            return observation, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def recenter(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        return self.env.ego_xy(pos)

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position, 3 dim or 2 dim '''
        return self.env.dist_xy(pos)

    def get_obs(self):
        '''
        We will ingnore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos
        vases_pos_list = self.env.vases_pos # list of shape (3,) ndarray
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray
        gremlins_pos_list = self.env.gremlins_obj_pos # list of shape (3,) ndarray
        buttons_pos_list = self.env.buttons_pos # list of shape (3,) ndarray

        ego_goal_pos = self.recenter(goal_pos[:2])
        ego_vases_pos_list = [self.env.ego_xy(pos[:2]) for pos in vases_pos_list] # list of shape (2,) ndarray
        ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray
        ego_gremlins_pos_list = [self.env.ego_xy(pos[:2]) for pos in gremlins_pos_list] # list of shape (2,) ndarray
        ego_buttons_pos_list = [self.env.ego_xy(pos[:2]) for pos in buttons_pos_list] # list of shape (2,) ndarray
        
        #ego_goal_dist = np.array([self.dist_xy(goal_pos)])
        #ego_vases_dist_list = np.array( [self.dist_xy(pos) for pos in vases_pos_list] )
        #ego_hazards_dist_list = np.array( [self.dist_xy(pos) for pos in hazards_pos_list] )

        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            if sensor=='accelerometer':
                obs[sensor] = self.env.world.get_sensor(sensor)[:1] # only x axis matters
            elif sensor=='ballquat_rear':
                obs[sensor] = self.env.world.get_sensor(sensor)
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)[:2] # only x,y axis matters

        for sensor in self.angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self.env.world.get_sensor(sensor)[2:] #[2:] # only z axis matters
                #pass # gyro does not help
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)

        obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
        obs["hazards"] = np.array(ego_hazards_pos_list) # (hazard_num, 2)
        obs["goal"] = ego_goal_pos # (2,)
        obs["gremlins"] = np.array(ego_gremlins_pos_list) # (vase_num, 2)
        obs["buttons"] = np.array(ego_buttons_pos_list) # (hazard_num, 2)

        # obs["goal_dist"] = ego_goal_dist
        # obs["vases_dist"] = ego_vases_dist_list
        # obs["hazards_dist"] = ego_hazards_dist_list
        return obs

    def get_obs_flatten(self):
        # get the flattened obs
        self.obs = self.get_obs()
        #obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        flat_obs = np.zeros(self.obs_flat_size)
        for k in self.flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = self.obs[k].flat
        return flat_obs

    def get_dist_reward(self):
        '''
        @return reward: negative distance from robot to the goal
        '''
        return -self.env.dist_goal()

    def calculate_target_control(self, action):
        '''
        calculate the target actuator states based on given action
        @param action [ndarray, (vx, w)]
        @return targets [ndarray, dimension depends on the robot actuator number]
        '''
        if self.robot == "Car":
            v_x, w = action
            R = 1 # robot radiud
            v_r = v_x + 0.5*R*w # right wheel speed
            v_l = v_x - 0.5*R*w # left wheel speed
            targets = np.array([v_r, v_l])
        elif self.robot == "Point":
            targets = action
        else:
            targets = action
        return targets

    def PID_controller(self, targets, observation):
        '''
        Low-level feedback controller.
        @param targets [ndarray]: target control output
        @param observation [ndarray]: observation of the target actuator speed from the sensors
        '''
        kp, ki = 1, 1
        error = observation - targets
        control = kp*error
        return control

    @property
    def observation_size(self):
        return self.state_dim

    # @property
    # def observation_space(self):
    #     return np.zeros(self.state_dim)

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self.env.action_space.low[0]), float(self.env.action_space.high[0])

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self.env.action_space.sample()



class EnvWrapper:
    def __init__(self, env):
        '''
        In PointGoal1 env, there will be 1 vase and 8 hazards.
        In PointGoal2 env, there will be 10 vases and 10 hazards.
        '''
        self.env = env
        self.env_name = env.unwrapped.spec.id
        self.robot_name = re.findall('[A-Z][a-z]*', self.env_name.split("-")[1])[0]
        print(self.robot_name)

        self.xyz_sensors = []
        self.angle_sensors = []
        self.vel_sensors = []

        if self.robot_name == 'Point':
            self.xyz_sensors = ['accelerometer', 'velocimeter']
            self.angle_sensors = []

        if self.robot_name == 'Car':
            self.xyz_sensors = ['velocimeter', 'accelerometer' ]#,'ballquat_rear']#, 'accelerometer']
            self.angle_sensors = ['magnetometer', 'gyro']#, 'ballangvel_rear']
            self.vel_sensors = ['right_wheel_vel', 'left_wheel_vel']

        self.thres = {"vases":1, "hazards":4}
        # obs is composed of constraints and base obs
        self.constraints_name = ["vases", "hazards"]
        self.base_state_name = self.xyz_sensors + self.angle_sensors + self.vel_sensors

        self.flatten_order = self.base_state_name + ["goal"] + self.constraints_name

        # get state space vector size
        self.env.reset()
        obs = self.get_obs()
        self.obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        self.state_dim = self.obs_flat_size
        self.key_to_slice = {}
        offset = 0
        for k in self.flatten_order:
            # print(obs[k])
            # print(obs[k].shape)
            k_size = np.prod(obs[k].shape)
            self.key_to_slice[k] = slice(offset, offset + k_size)
            print("obs key: ", k, " slice: ", self.key_to_slice[k])
            offset += k_size

        self.base_state_dim = sum([np.prod(obs[k].shape) for k in self.base_state_name])
        self.action_dim = self.env.action_space.shape[0]
        self.key_to_slice["base_state"] = slice(0, self.base_state_dim)

    def recenter(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        return self.env.ego_xy(pos)

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position '''
        return self.env.dist_xy(pos)

    def get_obs(self):
        '''
        We will ingnore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos
        vases_pos_list = self.env.vases_pos # list of shape (3,) ndarray
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray

        ego_goal_pos = self.recenter(goal_pos[:2])
        ego_vases_pos_list = [self.env.ego_xy(pos[:2]) for pos in vases_pos_list] # list of shape (2,) ndarray
        ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray

        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            if sensor=='accelerometer':
                obs[sensor] = self.env.world.get_sensor(sensor) # only x axis matters
            elif sensor=='ballquat_rear':
                obs[sensor] = self.env.world.get_sensor(sensor)
            else:
                obs[sensor] = self.env.world.get_sensor(sensor) # only x,y axis matters

        for sensor in self.angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self.env.world.get_sensor(sensor)
                #pass # gyro does not help
            elif sensor=='magnetometer':
                obs[sensor] = self.env.world.get_sensor(sensor)
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)

        for sensor in self.vel_sensors:
            obs[sensor] = self.env.world.get_sensor(sensor)

        obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
        obs["hazards"] = np.array(ego_hazards_pos_list) # (hazard_num, 2)
        obs["goal"] = ego_goal_pos # (2,)
        return obs

    def get_obs_flatten(self):
        # get the flattened obs
        self.obs = self.get_obs()
        #obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        flat_obs = np.zeros(self.obs_flat_size)
        for k in self.flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = self.obs[k].flat
        return flat_obs

    def flatten_obs(self, obs):
        # flatten the given obs dictionary
        obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        flat_obs = np.zeros(obs_flat_size)
        offset = 0
        for k in self.flatten_order:
            k_size = np.prod(obs[k].shape)
            flat_obs[offset:offset + k_size] = obs[k].flat
            offset += k_size
        return flat_obs

    def get_closest_pos(self, pos_list, n):
        '''
        Get top n closest poses in pos_list, and return the list of egocentric XY vector in the order of from small to large distance.

        @param pos_list [list of ndarray with shape (3,)]
        '''
        assert n<=len(pos_list), "no enough poses to return."
        # sort by the distance to ego robot
        sorted_list = sorted(pos_list, key=lambda pos:self.dist_xy(pos)) 
        # get the egocentric XY coordinates list for the top n closest.
        ego_pos_list = []
        for pos in sorted_list[:n]:
            ego_pos_list.append(self.env.ego_xy(pos[:2]))     
        return ego_pos_list

    def get_dist_reward(self):
        '''
        @return reward: negative distance from robot to the goal
        '''
        return -self.env.dist_goal()


    def relative_angle(self, pos):
        '''
        Return a robot-centric compass observation of a list of positions.

        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        '''
        return self.env.obs_compass(pos)

    def get_obs_with_ranked_distance(self):
        '''
        We will ingnore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos
        vases_pos_list = self.env.vases_pos # list of shape (3,) ndarray
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray

        ego_goal_pos = self.recenter(goal_pos[:2])
        ego_vases_pos_list = self.get_closest_pos(vases_pos_list, n=self.thres["vases"]) # list of shape (2,) ndarray
        ego_hazards_pos_list = self.get_closest_pos(hazards_pos_list, n=self.thres["hazards"]) # list of shape (2,) ndarray

        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            obs[sensor] = self.env.world.get_sensor(sensor)[:2] # only x y axis matters
        for sensor in self.angle_sensors:
            obs[sensor] = self.env.world.get_sensor(sensor)[2:] # only z axis matters

        obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
        obs["hazards"] = np.array(ego_hazards_pos_list) # (hazard_num, 2)
        obs["goal"] = ego_goal_pos # (2,)
        return obs