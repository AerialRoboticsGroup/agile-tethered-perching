from gym_pybullet_drones.main.algorithms.lr_schedular import LinearLearningRateSchedule
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import argparse
import numpy as np
import glob
import os
import time

from gym_pybullet_drones.utils.utils import sync, print_green, print_red, load_json, StoreDict
from gym_pybullet_drones.main.algorithms.sacfd import SACfD
from gym_pybullet_drones.main.algorithms.dual_buffer import DualReplayBuffer
from stable_baselines3 import SAC

from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv

from gym_pybullet_drones.main.algorithms.wrappers.position_wrapper import PositionWrapper
from gym_pybullet_drones.main.algorithms.wrappers.symmetric_wrapper import SymmetricWrapper
from gym_pybullet_drones.main.algorithms.wrappers.hovering_wrapper import HoveringWrapper


from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.main.algorithms.CustomEvalCallback import CustomEvalCallback

from gym_pybullet_drones.main.algorithms.TensorboardCallback import TensorboardCallback
    

DEMO_PATH = "../demonstration/training_demos_json"


# ---------------------------------- RL UTIL ----------------------------------

# Shows the demonstration data in the enviornment - useful for verification purpose
def show_in_env(env, transformed_data):
    '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
    """Display the demonstration data in the environment."""
    done = False
    start_time = time.time()

    for i, (state, next_obs, action, reward1, done_flag, info) in enumerate(transformed_data):
       
        obs, reward, done, truncated, _ = env.step(action)
        
        # print(f"State: {next_obs}, Simulated State: {obs}")
        
        env.render()

        sync(i, start_time, env.get_wrapper_attr('CTRL_TIMESTEP'))
        
        if done or truncated:
            print("1Episode finished")
            break
        
            
        
# ----------------------------------- DATA ------------------------------------

def get_buffer_data(env, directory, show_demos_in_env):
    """Load the demonstration data from JSON files and optionally display it in the environment."""
    '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
    pattern = f"{directory}/rl_demo_new_*.json"
    files = glob.glob(pattern)
    all_data = []
    
    for file in files:
        print(file)
        json_data = load_json(file)
        
        transformed_data = convert_data(env, json_data)
        
        if show_demos_in_env:
            
            position = transformed_data[0][0][:3]
            # position[2] = 3.0
            env.reset(position=position, variable=False)
            show_in_env(env, transformed_data)
            
        
        all_data.extend(transformed_data)
    
    return all_data


def convert_data(env, json_data):
    '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
    dataset = []
    num = 0
    for item in json_data:
       
        x, y, z, t = item['state']
        obs = np.append(np.array([x, y, z, t]), num / 100.0)
        
        
        _next_obs = item['next_state']
        x, y, z, t = _next_obs
        next_obs = np.append(np.array(np.array([x, y, z, t])), (num + 1) / 300.0)
        
        action = np.array(item['action'])

        reward = np.array(item['reward'])
        done = np.array([False])
        info = [{}]
        dataset.append((obs, next_obs, action, reward, done, info))
        num = num + 1

    for _ in range(1):  # Adds an extra action on the end which helps with wrapping.
        dataset.append((next_obs, next_obs, next_obs[:3], reward, done, info))
    dataset.append((next_obs, next_obs, next_obs[:3], reward, np.array([True]), info))
    return dataset

# ---------------------------- ENVIRONMENT & AGENT ----------------------------


def get_agent(algorithm, env, demo_path, show_demos_in_env, hyperparams, filename):
    

    _policy = "MlpPolicy"
    _seed = 0
    _batch_size = hyperparams.get("batch_size", 64)
    _policy_kwargs = dict(net_arch=[128, 128, 64])
    _lr_schedular = LinearLearningRateSchedule(hyperparams.get("lr", 0.001))
    _lr_schedular_sacfD = LinearLearningRateSchedule(hyperparams.get("lr", 0.001))
    # _lr_schedular = 3e-4
    
    print_green(f"Hyperparamters: seed={_seed}, batch_size={_batch_size}, policy_kwargs={_policy_kwargs}, " + (
                f"lr={_lr_schedular}"))

    if algorithm == "SAC":
        agent = SAC(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            learning_rate=_lr_schedular,
            gamma=0.96,
            policy_kwargs=_policy_kwargs,
            tensorboard_log= "./logs/",
            verbose=1
        )
    elif algorithm == "SACfD":
        agent = SACfD(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            policy_kwargs=_policy_kwargs,
            learning_starts=0,
            gamma=0.96,
            learning_rate=_lr_schedular_sacfD,
            replay_buffer_class=DualReplayBuffer,
            tensorboard_log= "./logs/",
            verbose = 1
        )
        pre_train(agent, env, demo_path, show_demos_in_env)

    else:
        print_red("ERROR: Not yet implemented",)
    return agent

def pre_train(agent, env, demo_path, show_demos_in_env):
    from stable_baselines3.common.logger import configure

    data = get_buffer_data(env, demo_path, show_demos_in_env)
    # print(data[-1])
    print("Buffer Size in pre_train(): ", agent.replay_buffer.size())

    for i in range(5):
        for obs, next_obs, action, reward, done, info in data:
            agent.replay_buffer._add_offline(obs, next_obs, action, reward, done, info)
    print("Online Buffer Size: ", agent.replay_buffer.online_replay_buffer.size())
    print("Offline Buffer Size: ", agent.replay_buffer.offline_replay_buffer.size())
    print_green("Pretraining Complete!")


# ----------------------------------- MAIN ------------------------------------

def test_agent(agent, env, num_episodes=5):
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = np.array(obs) 
        start = time.time()
        done = False
        total_reward = 0
        counter = 0
        while not done or not truncated:
            
            action, _states = agent.predict(obs, deterministic=True)
      
            obs, reward, done, truncated, info = env.step(action)
            
            # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", done, "\tTruncated:", truncated)
        
            payload_position = env.get_wrapper_attr('weight').get_position()
            
            total_reward += reward
            env.render()
            sync(counter, start, env.get_wrapper_attr('CTRL_TIMESTEP'))
            if done or truncated:
                break
            counter += 1
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
    env.close()
    
   
def main(algorithm, timesteps, demo_path, should_show_demo , hyperparams):
    
    #output folder named models
    filename = os.path.join("models/", 'save-algorithm-'+ algorithm +'-' +datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+'-'+ str(timesteps))
    
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
 
    train_env = Monitor(HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(gui=should_show_demo)))), filename)

    
    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    ##########################################################################
    #################### Train and evaluate the model ########################
    ##########################################################################
    
    agent = get_agent(algorithm, train_env, demo_path, should_show_demo, hyperparams, filename)
    
    
    ###############################################################################
    ###############################################################################
    ############ Evaluate in eval episodes - slower but more consistent ###########
    ###############################################################################
    ###############################################################################
    
    
    ## Could be better if we can create a seperate eval env - technical error - modelling
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=900,verbose=1)
    eval_callback = CustomEvalCallback(train_env,
                                 verbose=1,
                                #  callback_on_new_best=callback_on_best,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1500),
                                 deterministic=True,
                                 render=False,
                                 render_train = False,
                                 plot_rewards=False)
    
    agent.learn(timesteps, log_interval=10, progress_bar=True, callback=[eval_callback, TensorboardCallback()], tb_log_name= algorithm + "_training_" + str(timesteps))
    
    # agent.learn(timesteps, log_interval=10, progress_bar=True)
    print_green("TRAINING COMPLETE!")

    # train_env.reset()
    train_env.close()
    
    # eval_env.close()
    
    #### Save the model ########################################
    agent.save(filename+'/final_model.zip')
    print(f"model is saved to: {filename}")

    #### Print Evaluation progression ############################
    with np.load(filename + '/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results']  # Assuming this is an array of episode rewards per evaluation
        
        print(f"{'Timestep':<10} {'Mean Reward':<15} {'Std Dev (if any)':<15}")
        print("-" * 40)
        
        for j in range(timesteps.shape[0]):
            mean_reward = np.mean(results[j])
            std_reward = np.std(results[j]) if len(results[j]) > 1 else "N/A"  # Check if there's more than one episode
            print(f"{timesteps[j]:<10} {mean_reward:<15.2f} {std_reward:<15}")


    # ##########################################################################
    ###################### Load the trained model ############################
    ##########################################################################

    print_green("Press Enter to continue Testing...")
    input()
    # #########save the best model after evaluation
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        if os.path.isfile(filename+'/final_model.zip'):
            path = filename+'/final_model.zip'
    

    if algorithm == 'SACfD':
        model = SACfD.load(path)
    elif algorithm == 'SAC':
        model = SAC.load(path)
    else:
        print("[ERROR]: no model under the current algotithm", algorithm)
        
    ##########################################################################
    ###################### Show the model's performance ######################
    ##########################################################################
    
    print_green("Evaluating the Performance of the Trained Model...")
    test_env_nogui = Monitor(HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv( gui=True)))))
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=5
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    test_env_nogui.close()
    
    print_green("Testing the Performance of the Trained Model...")
    test_env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(gui=True))))
    test_agent(model, test_env)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training for Tethered Drone Perching")

    # Number of timesteps
    parser.add_argument('-t', '--timesteps', type=int, required=True,
                        help='Number of timesteps for training (e.g., 40000)')

    # Choice of algorithm
    parser.add_argument('-algo', '--algorithm', type=str, choices=['SAC', 'SACfD'], required=True,
                        help='Choice of algorithm: SAC or SACfD')

    # Demonstration path
    parser.add_argument('--demo-path', type=str, default=DEMO_PATH,
                        help=f"Path to demonstration files (default: {DEMO_PATH}")

    # Show demonstrations in visual environment
    parser.add_argument('--show-demo', action='store_true', help='Show visualisation in visual environment')

    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. lr:0.01 batch_size:10)",)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    algorithm = args.algorithm
    timesteps = args.timesteps
    demo_path = args.demo_path
    should_show_demo = args.show_demo

    if algorithm != "SACfD" and demo_path is not None:
        print_red("WARNING: Demo path provided will NOT be used by this algorithm!")

    print_green(f"Algorithm: {algorithm}")
    print_green(f"Timesteps: {timesteps}")

    if algorithm == "SACfD":
        print_green(f"Demo Path: {demo_path}")

    accpetable_hp = ["lr", "batch_size"]
    hyperparams = args.hyperparams if args.hyperparams is not None else dict()
    for key, val in hyperparams.items():
        if key in accpetable_hp:
            print_green(f"\t{key}: {val}")
        else:
            print_red(f"\nUnknown Hyperparameter: {key}")

    main(algorithm, timesteps, demo_path, should_show_demo , hyperparams)
