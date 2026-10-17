

#==================================================================================================
# Dependencies
#==================================================================================================

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*_dump_logs\(\) is deprecated.*",
    category=UserWarning,
)

import numpy as np
import torch as th
import pandas as pd
from os.path import join
import os
import gymnasium as gym
from gymnasium import spaces
from typing import TypeVar, Any, Callable, Optional, Union
import warnings
import json
import bz2
import base64
import numpy as np
import os
import time
import csv
from torch.nn import functional as F
import torch.nn as nn
import shutil
import sys
import math
import matplotlib.pyplot as plt

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.type_aliases import  MaybeCallback, TrainFreq
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")
SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

#==================================================================================================
# General classes
#==================================================================================================

class MyTimer:
    '''To monitor computation times throughout a learning process.'''
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t

def compress_decompress_list(my_list,compress=True):
    '''
    To compress lists that store data from an interaction (e.g., initial states, state sequences,
    reward sequences, gift sequences, etc.) and to make the databases that store this data per interaction more compact.
    '''
    if compress:
        # Convert the list to a compact JSON string.
        json_str = json.dumps(my_list)

        # Compress the string.
        compressed_data = bz2.compress(json_str.encode('utf-8'))

        # Convert to base64 to store it as text in the database.
        compressed_str = base64.b64encode(compressed_data).decode('utf-8')

        return compressed_str
    else:
        # Read the compressed string from the database.
        compressed_data = base64.b64decode(my_list.encode('utf-8'))

        # Decompress the string.
        json_str = bz2.decompress(compressed_data).decode('utf-8')

        # Convert the JSON string back to a list.
        my_list = json.loads(json_str)

        return my_list

#==================================================================================================
# Modified functions
#==================================================================================================

class ModifiedFunctions_Common:
    '''
    Changes to the `EvalCallback._on_step` default function have been added. This function defines the validation interaction. 
    It has been modified to allow for the storage of validation data (both data associated with validation-based criteria and ground truth data).
    Changes made are highlighted with a comment such as: # MODIFICATION.
    '''
    # Las modificaciones aplicadas permiten almacenar los datos necesarios al aplicar last, validation y cost-driven validation.
    # Los datos para train se completan parcialmente en este funcion, y terminan de completarse con las modificaciones de la funcion dump_logs
    def _on_step(self) -> bool:

        # MODIFICATION ############################################################################
        global all_initial_states, eval_env_name, make_eval_deterministic,make_normalize, n_envs_for_truth_eval, n_policy, make_vec_env_type, make_eval_for_test
        global best_policy,select_policy,best_reward,select_reward, select_est_reward, total_steps, criterion, criterion_conf, last_truth_reward
        global criterion_kappa, criterion_phi, criterion_phi_c
        global est_time_val_kappa, est_val_ep, total_val_time,cost_driven_validation_done

        cost_driven_validation_done=False

        def my_evaluate_policy(
            model: "type_aliases.PolicyPredictor",
            env: Union[gym.Env, VecEnv],
            n_eval_episodes: int = 10,
            deterministic: bool = True,
            render: bool = False,
            callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
            reward_threshold: Optional[float] = None,
            return_episode_rewards: bool = False,
            warn: bool = True,
        ) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
            """
            Runs policy for ``n_eval_episodes`` episodes and returns average reward.
            If a vector env is passed in, this divides the episodes to evaluate onto the
            different elements of the vector env. This static division of work is done to
            remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
            details and discussion.

            .. note::
                If environment has not been wrapped with ``Monitor`` wrapper, reward and
                episode lengths are counted as it appears with ``env.step`` calls. If
                the environment contains wrappers that modify rewards or episode lengths
                (e.g. reward scaling, early episode reset), these will affect the evaluation
                results as well. You can avoid this by wrapping environment with ``Monitor``
                wrapper before anything else.

            :param model: The RL agent you want to evaluate. This can be any object
                that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
                or policy (``BasePolicy``).
            :param env: The gym environment or ``VecEnv`` environment.
            :param n_eval_episodes: Number of episode to evaluate the agent
            :param deterministic: Whether to use deterministic or stochastic actions
            :param render: Whether to render the environment or not
            :param callback: callback function to do additional checks,
                called after each step. Gets locals() and globals() passed as parameters.
            :param reward_threshold: Minimum expected reward per episode,
                this will raise an error if the performance is not met
            :param return_episode_rewards: If True, a list of rewards and episode lengths
                per episode will be returned instead of the mean.
            :param warn: If True (default), warns user about lack of a Monitor wrapper in the
                evaluation environment.
            :return: Mean reward per episode, std of reward per episode.
                Returns ([float], [int]) when ``return_episode_rewards`` is True, first
                list containing per-episode rewards and second containing per-episode lengths
                (in number of steps).
            """
            is_monitor_wrapped = False
            # Avoid circular import
            from stable_baselines3.common.monitor import Monitor

            if make_eval_deterministic:# MODIFICATION.

                if make_eval_for_test:

                    if make_vec_env_type=='sequential':
                        env = make_vec_env(eval_env_name, n_envs=env.num_envs)
                    if make_vec_env_type=='parallel':
                        env = make_vec_env(eval_env_name, n_envs=env.num_envs,vec_env_cls=SubprocVecEnv)

                if not make_eval_for_test:

                    if make_vec_env_type=='sequential':
                        env = make_vec_env(eval_env_name, n_envs=n_envs_for_truth_eval)
                    if make_vec_env_type=='parallel':
                        env = make_vec_env(eval_env_name, n_envs=n_envs_for_truth_eval,vec_env_cls=SubprocVecEnv)

            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

            if make_normalize: #MODIFICATION: for when the parameter is set to normalize=true.
                env = VecNormalize(env, training=False, norm_reward=False)
                sync_envs_normalization(self.training_env, env)


            if make_eval_deterministic:# MODIFICATION: to make it deterministic and to consider different initial states for validation and ground truth.
                if make_eval_for_test: 
                    env.seed(99999)
                else:
                    env.seed(0)

            is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

            if not is_monitor_wrapped and warn:
                warnings.warn(
                    "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                    "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                    "Consider wrapping environment first with ``Monitor`` wrapper.",
                    UserWarning,
                )

            n_envs = env.num_envs
            episode_rewards = []
            episode_lengths = []
            episode_inits=[]# MODIFICATION.
            start_val_time=time.time()# MODIFICATION.
            num_episodes=[]# MODIFICATION.
            times_per_episode=[]# MODIFICATION.
            episode_inits_per_env=[[] for i in range(env.num_envs)]# MODIFICATION.



            episode_counts = np.zeros(n_envs, dtype="int")
            # Divides episodes among different sub environments in the vector as evenly as possible
            episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

            current_rewards = np.zeros(n_envs)
            current_lengths = np.zeros(n_envs, dtype="int")
            observations = env.reset()

            for i in range(env.num_envs):# MODIFICATION.
                episode_inits_per_env[i].append(observations[i])
 

            states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            while (episode_counts < episode_count_targets).any():
                actions, states = model.predict(
                    observations,  # type: ignore[arg-type]
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                new_observations, rewards, dones, infos = env.step(actions)
                current_rewards += rewards
                current_lengths += 1
                for i in range(n_envs):
                    if episode_counts[i] < episode_count_targets[i]:
                        # unpack values so that the callback can access the local variables
                        reward = rewards[i]
                        done = dones[i]
                        info = infos[i]
                        episode_starts[i] = done

                        if callback is not None:
                            callback(locals(), globals())

                        if dones[i]:
                            if is_monitor_wrapped:
                                # Atari wrapper can send a "done" signal when
                                # the agent loses a life, but it does not correspond
                                # to the true end of episode
                                if "episode" in info.keys():
                                    # Do not trust "done" with episode endings.
                                    # Monitor wrapper includes "episode" key in info if environment
                                    # has been wrapped with it. Use those rewards instead.
                                    episode_rewards.append(info["episode"]["r"])
                                    episode_lengths.append(info["episode"]["l"])
                                    # Only increment at the real end of an episode
                                    episode_counts[i] += 1
                                    num_episodes.append(sum(episode_counts))# MODIFICATION.
                                    times_per_episode.append(time.time()-start_val_time)# MODIFICATION.
                            else:
                                episode_rewards.append(current_rewards[i])
                                episode_lengths.append(current_lengths[i])
                                episode_counts[i] += 1
                                num_episodes.append(sum(episode_counts))# MODIFICATION.
                                times_per_episode.append(time.time()-start_val_time)# MODIFICATION.
                            current_rewards[i] = 0
                            current_lengths[i] = 0

                            episode_inits.append(compress_decompress_list(episode_inits_per_env[i][-1].tolist()))# MODIFICATION.
                            episode_inits_per_env[i].append(new_observations[i])# MODIFICATION.

                observations = new_observations

                if render:
                    env.render()

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            if reward_threshold is not None:
                assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
            if return_episode_rewards:
                return episode_rewards, episode_lengths, episode_inits, num_episodes, times_per_episode # MODIFICATION.
            return mean_reward, std_reward

        ###########################################################################################


        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []


            # MODIFICATION: First, validate to obtain the ground truth data.
            np_state = np.random.get_state()#Esto de aqui es para guardar el estado de aleatoriedad de numpy, ya que cuando criterion='validation'
                                            #se hacen dos llamadas a my_evaluate_policy, y el estado como cambia, despues los minibatches se escogen 
                                            #diferente para la actualizacion de la politica aunque sea el mismo rollout y se genera una secuencia de
                                            #politicas diferente. Nosotros queremos generar la misma secuencia para el mismo metodo-entorno-semilla,
                                            #aunque el criterio sea diferente
            make_eval_for_test=False
            episode_rewards_truth, episode_lengths_truth, episode_inits_truth, num_episodes_truth, times_per_episode_truth = my_evaluate_policy( # MODIFICATION.
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            # MODIFICATION: then validation for validation and cost-driven validation criteria simulation.
            if criterion in ['validation']:

                if n_policy%criterion_phi==0:
                    make_eval_for_test=True
                    episode_rewards, episode_lengths, episode_inits, num_episodes, times_per_episode = my_evaluate_policy( # MODIFICATION.
                        self.model,
                        self.eval_env,
                        n_eval_episodes=criterion_kappa,
                        render=self.render,
                        deterministic=self.deterministic,
                        return_episode_rewards=True,
                        warn=self.warn,
                        callback=self._log_success_callback,
                    )
                else:
                    times_per_episode=[0]

            if criterion in ['cost_driven_validation']:

                if (total_val_time+est_time_val_kappa)/total_time_seconds.get_time()<=criterion_phi_c:

                    make_eval_for_test=True
                    episode_rewards, episode_lengths, episode_inits, num_episodes, times_per_episode = my_evaluate_policy( # MODIFICATION.
                        self.model,
                        self.eval_env,
                        n_eval_episodes=criterion_kappa,
                        render=self.render,
                        deterministic=self.deterministic,
                        return_episode_rewards=True,
                        warn=self.warn,
                        callback=self._log_success_callback,
                    )
                    cost_driven_validation_done=True
                else:
                    times_per_episode=[0]

            np.random.set_state(np_state)

            # MODIFICATION ########################################################################
            # Write validation data of the new policy.

            last_truth_reward=np.mean(episode_rewards_truth)

            with open(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), 'a',newline='') as df_process:

                # Para saber cual es la mejor politica real
                if last_truth_reward>best_reward:
                    best_reward=last_truth_reward
                    best_policy=n_policy

                writer = csv.writer(df_process)

                if criterion=='last':
                    writer.writerow([n_policy,total_steps,total_time_seconds.get_time(),0,n_policy,last_truth_reward,best_policy,best_reward,last_truth_reward])

                if criterion=='train':
                    writer.writerow([n_policy,total_steps,total_time_seconds.get_time(),0,None,None,best_policy,best_reward,last_truth_reward])

                if criterion=='validation':

                    if n_policy%criterion_phi==0:
                        if np.mean(episode_rewards)>select_est_reward:
                            select_policy=n_policy
                            select_reward=last_truth_reward
                            select_est_reward=np.mean(episode_rewards)
                        

                    writer.writerow([n_policy,total_steps,total_time_seconds.get_time(),sum(times_per_episode),select_policy,select_reward,best_policy,best_reward,last_truth_reward])

                if criterion=='cost_driven_validation':

                    if cost_driven_validation_done:

                        # Actualizacion de la mejor politica
                        if np.mean(episode_rewards)>select_est_reward:
                            select_policy=n_policy
                            select_reward=last_truth_reward
                            select_est_reward=np.mean(episode_rewards)

                        # Actualizacion de constantes
                        est_val_ep=max(est_val_ep,max(times_per_episode))
                        est_time_val_kappa=criterion_kappa*est_val_ep
                        total_val_time+=sum(times_per_episode)

                    writer.writerow([n_policy,total_steps,total_time_seconds.get_time(),sum(times_per_episode),select_policy,select_reward,best_policy,best_reward,last_truth_reward])

            #######################################################################################

            # if self.log_path is not None:
            #     assert isinstance(episode_rewards, list)
            #     assert isinstance(episode_lengths, list)
            #     self.evaluations_timesteps.append(self.num_timesteps)
            #     self.evaluations_results.append(episode_rewards)
            #     self.evaluations_length.append(episode_lengths)


            #     kwargs = {}
            #     # Save success log if present
            #     if len(self._is_success_buffer) > 0:
            #         self.evaluations_successes.append(self._is_success_buffer)
            #         kwargs = dict(successes=self.evaluations_successes)

            #     # MODIFICATION: This data is already stored in the databases, along with additional data that is stored by default using the callback function.
            #     # np.savez(
            #     #     self.log_path,
            #     #     timesteps=self.evaluations_timesteps,
            #     #     results=self.evaluations_results,
            #     #     ep_lengths=self.evaluations_length,
            #     #     **kwargs,  # type: ignore[arg-type]
            #     # )

            # mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            # mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # self.last_mean_reward = float(mean_reward)

            # if self.verbose >= 1:
            #     print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            #     print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # # Add to current Logger
            # self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_ep_length", mean_ep_length)

            # if len(self._is_success_buffer) > 0:
            #     success_rate = np.mean(self._is_success_buffer)
            #     if self.verbose >= 1:
            #         print(f"Success rate: {100 * success_rate:.2f}%")
            #     self.logger.record("eval/success_rate", success_rate)

            # # Dump log so the evaluation results are printed with the correct timestep
            # self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            # self.logger.dump(self.num_timesteps)

            # if mean_reward > self.best_mean_reward:
            #     if self.verbose >= 1:
            #         print("New best mean reward!")
            #     if self.best_model_save_path is not None:# MODIFICATION: When we store the data, we do not save the policies
            #         pass
            #         #self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            #     self.best_mean_reward = float(mean_reward)
            #     # Trigger callback on new best model, if needed
            #     if self.callback_on_new_best is not None:
            #         continue_training = self.callback_on_new_best.on_step()

            # # Trigger callback after every evaluation, if needed
            # if self.callback is not None:
            #     continue_training = continue_training and self._on_event()

        return continue_training

    
class ModifiedFunctions_OnPolicy:
    '''
    Changes added to the following functions:
    `OnPolicyAlgorithm.learn` (named as learn): This is the main learning function, and it has been modified to initialize the databases that will store the data for the simulations, allowing the possibility of also saving the policies that have been visited.
    `OnPolicyAlgorithm.collect_rollouts` (named as collect_rollouts): This function is involved in the train interaction and is modified to save the training interaction data.
    `PPO.train` (named as PPO_train): This function is involved in policy updates and is modified to save the policy data used for those updates.

    Changes made in each original function are highlighted with a comment such as: # MODIFICATION.
    '''

    # Funcion modificada para inicializar base de datos en que se almacenaran los datos relevantes del proceso, con la posibilidad de guardar las politicas visitadas
    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        
        # MODIFICATION ############################################################################
        global process_dir, n_policy, total_time_seconds, make_policy_saving 
        global best_policy,select_policy,best_reward,select_reward, select_est_reward, total_steps
        global criterion, criterion_conf
        global est_time_val_kappa, est_val_ep, total_val_time
        n_policy=0
        best_policy=0
        select_policy=0
        best_reward=-math.inf
        select_reward=-math.inf
        select_est_reward=-math.inf
        total_time_seconds=MyTimer()
        total_time_seconds.reset()
        est_val_ep=-math.inf
        est_time_val_kappa=0
        total_val_time=0
        

        total_time_seconds.pause()
        # Save initial policy.
        if make_policy_saving:
            self.save(process_dir+'/policy'+str(n_policy)+'.zip')



        # Create database where data will be written iteratively.
        df_process=pd.DataFrame(columns=['iter','n_timesteps','time_learn','time_val','indx_select','reward_select','indx_best','reward_best','reward_last'])
        df_process.to_csv(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), index=False)
        total_time_seconds.resume()
        ###########################################################################################

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:

            progress = min(self.num_timesteps / total_timesteps, 1.0)
            sys.stdout.write(f"\rCompleting learning process: {progress:.1%}")
            sys.stdout.flush()

            total_steps=self.num_timesteps # MODIFICATION

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()


            # MODIFICATION ########################################################################
            total_time_seconds.pause()
            
            # Save policy.
            if make_policy_saving:
                self.save(process_dir+'/policy'+str(n_policy)+'.zip')
            n_policy+=1
            total_time_seconds.resume()
            #######################################################################################

        callback.on_training_end()

        return self

    # TODO: esta funcion me puede servir ahora para el criterio train propuesto por el revisor: off-policy evaluation train (OPE)
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        # MODIFICATION ############################################################################
        global total_time_seconds, process_dir


        total_time_seconds.pause()
        global n_policy,df_traj
        policy_traj_rewards=[[] for _ in range(env.num_envs)]
        policy_traj_ep_end=[[] for _ in range(env.num_envs)]
        policy_traj_ep_inits=[[] for _ in range(env.num_envs)]
        total_time_seconds.resume()
        ###########################################################################################



        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            total_time_seconds.pause()# MODIFICATION: When a callback isn't defined, this takes no time at all, but when one is defined, it takes time because validation is performed.
            if not callback.on_step():
                return False
            total_time_seconds.resume()# MODIFICATION.

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # MODIFICATION ########################################################################
            total_time_seconds.pause()
            for i in range(env.num_envs):
                policy_traj_rewards[i].append(float(rewards[i]))
                policy_traj_ep_end[i].append(True if dones[i] else False)
                if policy_traj_ep_end[i][-1]:
                    policy_traj_ep_inits[i].append(compress_decompress_list(new_obs[i].tolist()))
            total_time_seconds.resume()
            #######################################################################################

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # MODIFICATION ########################################################################
        total_time_seconds.pause()
        with open(join(process_dir, "df_traj.csv"), 'a',newline='') as df_traj_csv:
            writer = csv.writer(df_traj_csv)
            # writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end),compress_decompress_list(policy_traj_ep_inits),
            #                  compress_decompress_list(rollout_buffer.advantages.tolist()),compress_decompress_list(rollout_buffer.values.tolist()),compress_decompress_list(rollout_buffer.returns.tolist()),None,None,None,None]) # NOTE: Uncomment this line and comment out the next one if you want to save the initial states of the trajectories.
            writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end),None,
                    compress_decompress_list(rollout_buffer.advantages.tolist()),compress_decompress_list(rollout_buffer.values.tolist()),compress_decompress_list(rollout_buffer.returns.tolist()),None,None,None,None])

        total_time_seconds.resume()
        ###########################################################################################

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    # TODO: esta funcion me puede servir ahora para el criterio test propuesto por el revisor: event-triggered validation
    def PPO_train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # MODIFICATION ############################################################################
        # Finish filling in the train iteration data by saving certain estimates associated with the policy, such as:
        # - policy_loss: the loss used to update the policy
        # - entropy_loss: measures the uncertainty/stochasticity of the policy
        # - KL_div: measures how different the current/updated policy is from the previous one
        global total_time_seconds
        total_time_seconds.pause()
        df_traj_csv = pd.read_csv(join(process_dir, "df_traj.csv"))
        
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('policy_loss')] = loss.item()
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('value_loss')] = np.mean(value_losses)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('entropy_loss')] = -np.mean(entropy_losses)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('policy_gradient_loss')] = np.mean(pg_losses)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('KL_div')] = np.mean(approx_kl_divs)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('explained_variance')] =  explained_var
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('log_std')] = th.exp(self.policy.log_std).mean().item()

        df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
        total_time_seconds.resume()

        ###########################################################################################
    
    # Funcion modificada para aplicar el best policy tracking usando el criterio train
    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

        # MODIFICATION ##############################################
        global select_policy,select_reward,select_est_reward, criterion, criterion_conf, last_truth_reward, n_policy

        if criterion=='train':

            # Comprobar si la mejor politica segun la estimacion del criterio a cambiado
            if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])>select_est_reward:
                select_policy=n_policy
                select_reward=last_truth_reward
                select_est_reward=safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

            # Completar las dos columnas de la ultima iteracion que faltaban por completar en el df
            with open(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), "r", newline="") as df_process:
                rows = list(csv.reader(df_process))
                rows[-1][4] = select_policy
                rows[-1][5] = select_reward 
    
            with open(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), "w", newline="") as df_process:
                writer = csv.writer(df_process)
                writer.writerows(rows)

        #############################################################

class ModifiedFunctions_OffPolicy:

    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:

        # MODIFICATION ############################################################################
        global process_dir, n_policy, total_time_seconds, make_policy_saving 
        global best_policy,select_policy,best_reward,select_reward, select_est_reward, total_steps
        global criterion, criterion_conf
        global est_time_val_kappa, est_val_ep, total_val_time
        n_policy=0
        best_policy=0
        select_policy=0
        best_reward=-math.inf
        select_reward=-math.inf
        select_est_reward=-math.inf
        total_time_seconds=MyTimer()
        total_time_seconds.reset()
        est_val_ep=-math.inf
        est_time_val_kappa=0
        total_val_time=0
        
        total_time_seconds.pause()
        # Save initial policy.
        if make_policy_saving:
            self.save(process_dir+'/policy'+str(n_policy)+'.zip')

        # Create database where data will be written iteratively.
        df_process=pd.DataFrame(columns=['iter','n_timesteps','time_learn','time_val','indx_select','reward_select','indx_best','reward_best','reward_last'])
        df_process.to_csv(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), index=False)
        total_time_seconds.resume()
        ###########################################################################################
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            progress = min(self.num_timesteps / total_timesteps, 1.0)
            sys.stdout.write(f"\rCompleting learning process: {progress:.4%}")
            sys.stdout.flush()

            total_steps=self.num_timesteps # MODIFICATION

            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:

                    # MODIFICATION ########################################################################
                    if criterion=='train':

                        # Comprobar si la mejor politica segun la estimacion del criterio a cambiado
                        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])>select_est_reward:
                            select_policy=n_policy
                            select_reward=last_truth_reward
                            select_est_reward=safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

                        # Completar las dos columnas de la ultima iteracion que faltaban por completar en el df
                        with open(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), "r", newline="") as df_process:
                            rows = list(csv.reader(df_process))
                            rows[-1][4] = select_policy
                            rows[-1][5] = select_reward 
                
                        with open(join(process_dir, "df_process_"+criterion+'_'+criterion_conf+".csv"), "w", newline="") as df_process:
                            writer = csv.writer(df_process)
                            writer.writerows(rows)
                    #######################################################################################

                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            # MODIFICATION ########################################################################
            total_time_seconds.pause()
            
            # Save policy.
            if make_policy_saving:
                self.save(process_dir+'/policy'+str(n_policy)+'.zip')
            n_policy+=1
            total_time_seconds.resume()
            #######################################################################################

        callback.on_training_end()

        return self


#==================================================================================================
#  New functions for generating learning processes
#==================================================================================================

class Options:

    def OnPolicy_learn_process(method,env_name,
                      seed,total_timesteps,experiment_name,library_dir, save_policies=True,
                      n_steps_per_env=2048,n_workers=1, truth_n_workers=1,
                      n_epoch=10,batch_size=64, 
                      device='auto', vec_env_type='sequential', 
                      normalize=False, 
                      policy='MlpPolicy',gae_lambda=0.95,gamma=0.99,ent_coef=0,learning_rate=0.0003,
                      clip_range=0.2,max_grad_norm=0.5, vf_coef=0.5,sde_sample_freq=-1,use_sde=False, policy_kwargs=None,
                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False,
                      selec_criterion={ "name": 'last', "kappa": None,"phi": None, "phi_c": None}):

        '''
        Training an on-policy reinforcement learning agent using Stable-baselines3.

        This function configures the environment, initializes the selected on-policy algorithm, 
        performs training, and stores the relevant data for each iteration so that different selection criteria
        can later be simulated on this same training process without having to rerun it for each different 
        criterion considered. To do this, two databases are stored in the `library_dir` directory: 
        `df_traj.csv` and `df_val.csv`, containing data per learning iteration associated with training 
        (for the train criterion) and validation (for criteria based on validation and ground truth). 
        
        Parameters are grouped into the following categories:

        Experiment configuration
        ........................
            `method`: RL method used for training is now programmed to run exclusively with PPO.
            `env_name`: environment name.
            `seed`: random seed.
            `total_timesteps`: number of total learning interaction steps that delimit the learning time.
            `experiment_name`: name of the folder where the process data will be stored.
            `library_dir`: path to the folder where the process data will be stored.
            `save_policies`: True or False depending on whether the models associated with the viewed policies should be saved in the specified path or not (default: True)

        Environment interaction
        .......................
            `n_steps_per_env`: the number of steps to run for each environment per update (each interaction duration).
            `n_workers`: number of copies of the environment considered in parallel for each interaction (dimension of vectorized environment), equivalent to the number of CPUs.
            `truth_n_workers`: considered parallelization to generate the ground truth data. For the ground truth, the number of workers can be set higher than the default number used for validation, since this will save time when storing the data (the time spent on the ground truth is not counted in the analysis).

        Policy update
        .............
            `n_epoch`: Number of epoch when optimizing the surrogate loss.
            `batch_size`: Minibatch size.

        Execution
        .........
            `device`: Device (cpu, cuda, ...) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
            `vec_env_type`: 'sequential' or 'parallel'

        Environment
        ...........
            `normalize`: normalization of the states of the environment for learning and validation (default: False).

        Learning hyperparameters
        ........................
            `policy`: The policy model to use (MlpPolicy, CnnPolicy, ...).
            `gae_lambda`: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
            `gamma`: Discount factor.
            `ent_coef`: Entropy coefficient for the loss calculation.
            `learning_rate`: The learning rate, it can be a function of the current progress remaining (from 1 to 0).
            `clip_range`: Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
            `max_grad_norm`: The maximum value for the gradient clipping.
            `vf_coef`: Value function coefficient for the loss calculation.
            `sde_sample_freq`: Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning of the rollout).
            `use_sde`: Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False).
            `policy_kwargs`: additional arguments to be passed to the policy on creation.


        For policy evaluation (related to selection criteria)
        .....................................................

            `callback`: Set it to True to validate the policies that have been visited.
            `n_eval_ep`: The number of episodes to validate the current policy.
            `eval_freq`: The period, expressed in the number of training steps, at which the callback function is called. Set `eval_freq = n_steps_per_env*n_workers` in order to store the validation data associated with all visited policies.
            `n_eval_envs`: Number of parallel environments considered for completing the `n_eval_ep` validation episodes. For a balanced use of parallelization in training and validation, it is recommended to set `n_eval_env = n_workers`.
            `deterministic_eval`: Set to True to make validation interactions deterministic, and use the same initial states for validating the different policies visited (this enables a paired comparison of policy rewards).
            `stats_window_size`: Window size for the rollout logging, specifying the number of episodes to average the reported success rate, mean episode length, and mean reward over (related to kappa parameter of train criteria). Currently does not affect data storage.

        The default values correspond to the Stable-Baselines3 defaults. Environment- and algorithm-specific hyperparameters can later be
        overridden according to the predefined configurations used in RL Baselines3 Zoo.
        '''

        
        # Global variables.
        global df_traj, process_dir,make_policy_saving, all_initial_states, eval_env_name, make_eval_deterministic, make_vec_env_type, n_envs_for_truth_eval, make_normalize
        
        df_traj=[]
        make_policy_saving=save_policies
        process_dir=library_dir+'/'+experiment_name+'/seed'+str(seed)
        all_initial_states=[]
        eval_env_name=env_name
        n_envs_for_truth_eval=truth_n_workers
        make_eval_deterministic=deterministic_eval
        make_vec_env_type=vec_env_type
        make_normalize=normalize

        global criterion, criterion_conf, criterion_kappa, criterion_phi, criterion_phi_c
        criterion=selec_criterion["name"]
        criterion_kappa=selec_criterion["kappa"]
        criterion_phi=selec_criterion["phi"]
        criterion_phi_c=selec_criterion["phi_c"]

        # Configuracion
        if criterion=='last':
            criterion_conf=''
        if criterion=='train':
            criterion_conf=str(selec_criterion["kappa"])
        if criterion=='validation':
            criterion_conf=str(selec_criterion["kappa"])+'_'+str(selec_criterion["phi"])
        if criterion=='cost_driven_validation':
                    criterion_conf=str(selec_criterion["kappa"])+'_'+str(selec_criterion["phi_c"])

        # Create new directories.
        os.makedirs(process_dir, exist_ok=True)


        # Modify functions in existing libraries.
        from stable_baselines3.ppo import MlpPolicy, PPO
        OnPolicyAlgorithm.learn=ModifiedFunctions_OnPolicy.learn
        OnPolicyAlgorithm.dump_logs=ModifiedFunctions_OnPolicy.dump_logs

        # TODO: estos dos descomentar cuando haya remodificado las funciones para aplicar los otros criterios train y validation propuestos por el revisor
        #OnPolicyAlgorithm.collect_rollouts=ModifiedFunctions_OnPolicy.collect_rollouts
        #PPO.train=ModifiedFunctions_OnPolicy.PPO_train

        if callback==True:
            EvalCallback._on_step= ModifiedFunctions_Common._on_step

        # Begin the learning process by setting the method, the environment, and the seed.

        # ----- Training environment

        if vec_env_type=='sequential':# By default, it is DummyVecEnv, which runs sequentially.
            env = make_vec_env(env_name, n_envs=n_workers)
        if vec_env_type=='parallel':
            env = make_vec_env(env_name, n_envs=n_workers,vec_env_cls=SubprocVecEnv)

        # ----- Validation environment
        if vec_env_type=='sequential':
            eval_env = make_vec_env(env_name, n_envs=n_eval_envs)
        if vec_env_type=='parallel':
            eval_env = make_vec_env(env_name, n_envs=n_eval_envs,vec_env_cls=SubprocVecEnv)

        # ----- In the case of normalization
        if normalize:
            env = VecNormalize(env)
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

        if method=='PPO':
            model = PPO(policy,
                        env, seed=seed,
                        n_steps=n_steps_per_env,batch_size=batch_size,n_epochs=n_epoch,
                        learning_rate=learning_rate,gamma=gamma,gae_lambda=gae_lambda,clip_range=clip_range,ent_coef=ent_coef,max_grad_norm=max_grad_norm,vf_coef=vf_coef,sde_sample_freq=sde_sample_freq,use_sde=use_sde,policy_kwargs=policy_kwargs,
                        stats_window_size=[selec_criterion['kappa'] if selec_criterion['kappa']!=None else 100][0],
                        verbose=0,device=device)

        model.set_random_seed(seed)
        

        # Esto lo hago para cualquier criterio, ya que es lo que nos permitira guardar los datos de truth
        # TODO: aunque deberia quizas ser una opcion optativa
        callback=EvalCallback(eval_env,n_eval_episodes=n_eval_ep,eval_freq=eval_freq,
                                  log_path=library_dir+'/'+experiment_name,best_model_save_path=library_dir+'/'+experiment_name,verbose=0)

        model.learn(total_timesteps=total_timesteps,callback=callback)

        
        sys.stdout.write("\rLearning process successfully completed!                    \n")
        sys.stdout.flush()

    def OffPolicy_learn_process(method,env_name,seed,total_timesteps,experiment_name,library_dir, save_policies=True, 
                      train_freq=1,n_workers=1,truth_n_workers=1, 
                      n_epoch=1,batch_size=256, 
                      device='auto', vec_env_type='sequential', 

                      # antes estos no los tenia
                      normalize=False, 
                      policy='MlpPolicy',

                      # TODO: es posible que aqui tenga que añadir mas parametros segun los entornos que incluya para SAC
                      tau=0.005,gamma=0.99,ent_coef='auto',learning_rate=0.0003,buffer_size=1000000,
                      learning_starts=100,use_sde=False,policy_kwargs=None,

                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False,#stats_window_size=100, 

                      # Antes este no lo tenia
                      selec_criterion={ "name": 'last', "kappa": None,"phi": None, "phi_c": None}

                      ):
        
        # Variables globales
        global df_traj, process_dir,make_policy_saving,all_initial_states, eval_env_name, make_eval_deterministic, make_vec_env_type, n_envs_for_truth_eval, make_normalize
        df_traj=[]
        make_policy_saving= save_policies
        process_dir=library_dir+'/'+experiment_name+'/seed'+str(seed)
        all_initial_states=[]
        eval_env_name=env_name
        n_envs_for_truth_eval=truth_n_workers
        make_eval_deterministic=deterministic_eval
        make_vec_env_type=vec_env_type
        make_normalize=normalize

        global criterion, criterion_conf, criterion_kappa, criterion_phi, criterion_phi_c
        criterion=selec_criterion["name"]
        criterion_kappa=selec_criterion["kappa"]
        criterion_phi=selec_criterion["phi"]
        criterion_phi_c=selec_criterion["phi_c"]

        # Configuracion
        if criterion=='last':
            criterion_conf=''
        if criterion=='train':
            criterion_conf=str(selec_criterion["kappa"])
        if criterion=='validation':
            criterion_conf=str(selec_criterion["kappa"])+'_'+str(selec_criterion["phi"])
        if criterion=='cost_driven_validation':
                    criterion_conf=str(selec_criterion["kappa"])+'_'+str(selec_criterion["phi_c"])


        # Crear nuevos directorios.
        os.makedirs(process_dir, exist_ok=True)

        # Modificar funciones de librerias existentes 
        from stable_baselines3.sac import MlpPolicy, SAC
        OffPolicyAlgorithm.learn=ModifiedFunctions_OffPolicy.learn

        # OffPolicyAlgorithm.collect_rollouts=ModifiedFunctions_OffPolicy.collect_rollouts
        # SAC.train=ModifiedFunctions_OffPolicy.SAC_train

        if callback==True:
            EvalCallback._on_step= ModifiedFunctions_Common._on_step

        # Begin the learning process by setting the method, the environment, and the seed.
        # ----- Training environment

        if vec_env_type=='sequential':# By default, it is DummyVecEnv, which runs sequentially.
            env = make_vec_env(env_name, n_envs=n_workers)
        if vec_env_type=='parallel':
            env = make_vec_env(env_name, n_envs=n_workers,vec_env_cls=SubprocVecEnv)

        # ----- Validation environment
        if vec_env_type=='sequential':
            eval_env = make_vec_env(env_name, n_envs=n_eval_envs)
        if vec_env_type=='parallel':
            eval_env = make_vec_env(env_name, n_envs=n_eval_envs,vec_env_cls=SubprocVecEnv)

        # ----- In the case of normalization
        if normalize:
            env = VecNormalize(env)
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

        if method=='SAC': 
            model = SAC(policy,
                    env, seed=seed,
                    train_freq=train_freq,batch_size=batch_size,gradient_steps=n_epoch,
                    # TODO: aqui habra que añadir las variables necesarias que modifiquen los parametros de aprendizaje en las conf zoo para SAC incluidas
                    tau=tau,gamma=gamma,ent_coef=ent_coef,learning_rate=learning_rate,buffer_size=buffer_size, learning_starts=learning_starts,
                    policy_kwargs=policy_kwargs,use_sde=use_sde,
                    stats_window_size=[selec_criterion['kappa'] if selec_criterion['kappa']!=None else 100][0],
                    verbose=0,device=device)

        model.set_random_seed(seed)

        # Esto lo hago para cualquier criterio, ya que es lo que nos permitira guardar los datos de truth
        # TODO: aunque deberia quizas ser una opcion optativa
        callback=EvalCallback(eval_env,n_eval_episodes=n_eval_ep,eval_freq=eval_freq,
                                  log_path=library_dir+'/'+experiment_name,best_model_save_path=library_dir+'/'+experiment_name,verbose=0)

        model.learn(total_timesteps=total_timesteps,callback=callback)

        
        sys.stdout.write("\rLearning process successfully completed!                    \n")
        sys.stdout.flush()


class PackOptions:
    '''
    Training an on-policy reinforcement learning agent using Stable-baselines3 with PPO, 
    and specific configurations determined by the method-environment pack in the RL Baselines3 Zoo.

    TODO: aqui creo que debo añadir como parametro modificable manualemente en cada una de las funciones
    el valor de n_eval_ep
    '''
    # On-policy packs (with PPO)
    #------------------------  Classic Control
    def PPO_Pendulum(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):

        ''' Pendulum-v1:
                n_envs: 4
                n_timesteps: !!float 1e5
                policy: 'MlpPolicy'
                n_steps: 1024
                gae_lambda: 0.95
                gamma: 0.9
                n_epochs: 10
                ent_coef: 0.0
                learning_rate: !!float 1e-3
                clip_range: 0.2
                use_sde: True
                sde_sample_freq: 4

        '''


        if available_resources<4:
            print('Insufficient resources: Running PPO and Pendulum in the default configuration requires 4 CPUs')
        else:
            print('Sufficient resources: The default execution with PPO and Pendulum uses only 4 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OnPolicy_learn_process(
                'PPO','Pendulum-v1', # pack
                seed,1e5+.5*1e5,'SB3_PPO_Pendulum',library_dir,save_policies=False, # learning process
                n_steps_per_env=1024,n_workers=4,truth_n_workers=available_resources, # learning interaction
                n_epoch=10, # policy update
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy',gae_lambda=0.95,gamma=0.9,ent_coef=0.0, learning_rate=1e-3,clip_range=0.2,
                use_sde=True,sde_sample_freq=4, # learning process parameters

                callback=True, n_eval_ep=n_eval_ep, eval_freq=1024, n_eval_envs=4, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    def PPO_MountailCarContinuous(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):

        ''' MountainCarContinuous-v0:
                normalize: true
                n_envs: 1
                n_timesteps: !!float 20000
                policy: 'MlpPolicy'
                batch_size: 256
                n_steps: 8
                gamma: 0.9999
                learning_rate: !!float 7.77e-05
                ent_coef: 0.00429
                clip_range: 0.1
                n_epochs: 10
                gae_lambda: 0.9
                max_grad_norm: 5
                vf_coef: 0.19
                use_sde: True
                policy_kwargs: "dict(log_std_init=-3.29, ortho_init=False)"

        '''


        if available_resources<1:
            print('Insufficient resources: Running PPO and MountainCarContinuous in the default configuration requires 1 CPUs')
        else:
            print('Sufficient resources: The default execution with PPO and MountainCarContinuous uses only 1 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OnPolicy_learn_process(
                'PPO','MountainCarContinuous-v0', # pack
                seed,20000+.5*20000,'SB3_PPO_MountainCarContinuous',library_dir,save_policies=False, # learning process
                n_steps_per_env=8,n_workers=1,truth_n_workers=available_resources, # learning interaction
                n_epoch=10, batch_size=256,# policy update
                device='auto', vec_env_type='sequential', normalize=True, # execution type
                policy='MlpPolicy',gae_lambda=0.9,gamma=0.9999,ent_coef=0.00429, learning_rate=7.77e-05,clip_range=0.1,
                max_grad_norm=5,use_sde=True,vf_coef=0.19, policy_kwargs=dict(log_std_init=-3.29, ortho_init=False), # learning process parameters
                callback=True, n_eval_ep=n_eval_ep, eval_freq=8, n_eval_envs=1, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    #------------------------  Box2D
    def PPO_LunarLanderContinuous(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):

        ''' Configuracion tomada de: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
            LunarLanderContinuous-v3:
            n_envs: 16
            n_timesteps: !!float 1e6
            policy: 'MlpPolicy'
            n_steps: 1024
            batch_size: 64
            gae_lambda: 0.98
            gamma: 0.999
            n_epochs: 4
            ent_coef: 0.01

        '''


        if available_resources<16:
            print('Insufficient resources: Running PPO and LunarLanderContinuos in the default configuration requires 16 CPUs')
        else:
            print('Sufficient resources: The default execution with PPO and LunarLanderContinuos uses only 16 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OnPolicy_learn_process(
                'PPO','LunarLanderContinuous-v3', # pack
                seed,1e6+.5*1e6,'SB3_PPO_LunarLanderContinuous',library_dir,save_policies=False, # learning process
                n_steps_per_env=1024,n_workers=16,truth_n_workers=available_resources, # learning interaction
                n_epoch=4,batch_size=64, # policy update
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy',gae_lambda=0.98,gamma=0.999,ent_coef=0.01, # learning process parameters
                callback=True, n_eval_ep=n_eval_ep, eval_freq=1024, n_eval_envs=16, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    def PPO_BipedalWalker(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):

        ''' Configuracion tomada de: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
        BipedalWalker-v3:
            normalize: true
            n_envs: 32
            n_timesteps: !!float 5e6
            policy: 'MlpPolicy'
            n_steps: 2048
            batch_size: 64
            gae_lambda: 0.95
            gamma: 0.999
            n_epochs: 10
            ent_coef: 0.0
            learning_rate: !!float 3e-4
            clip_range: 0.18

        '''

        if available_resources<32:
            print('Insufficient resources: Running PPO and BipedalWalker in the default configuration requires 32 CPUs')
        else:
            print('Sufficient resources: The default execution with PPO and BipedalWalker uses only 32 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OnPolicy_learn_process(
                'PPO','BipedalWalker-v3', # pack
                seed,5e6+.5*5e6,'SB3_PPO_BipedalWalker',library_dir,save_policies=False, # learning process
                n_steps_per_env=2048,n_workers=32, truth_n_workers=available_resources, normalize=True, # learning interaction
                n_epoch=10,batch_size=64, # policy update
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy',gae_lambda=0.95,gamma=0.999,ent_coef=0.0, learning_rate=3e-4, clip_range=0.18, # learning process parameters
                callback=True, n_eval_ep=n_eval_ep, eval_freq=2048, n_eval_envs=32, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    #------------------------  MuJoCo
    def PPO_Walker2d(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):

        ''' Configuracion tomada de: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
            Walker2d-v4:
                normalize: true
                n_envs: 1
                policy: 'MlpPolicy'
                n_timesteps: !!float 1e6
                batch_size: 32
                n_steps: 512
                gamma: 0.99
                learning_rate: 5.05041e-05
                ent_coef: 0.000585045
                clip_range: 0.1
                n_epochs: 20
                gae_lambda: 0.95
                max_grad_norm: 1
                vf_coef: 0.871923

        '''

        print('Sufficient resources: The default execution with PPO and Walker2d uses only 1 CPUs,')
        print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

        Options.OnPolicy_learn_process(
            'PPO','Walker2d-v4', # pack
            seed,1e6+.5*1e6,'SB3_PPO_Walker2d',library_dir,save_policies=False, # learning process
            n_steps_per_env=512,n_workers=1,truth_n_workers=available_resources, normalize=True, # learning interaction
            n_epoch=20,batch_size=32, # policy update
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gae_lambda=0.95,gamma=0.99,ent_coef=0.000585045, learning_rate=5.05041e-05, clip_range=0.1,max_grad_norm=1, vf_coef= 0.871923,# learning process parameters
            callback=True, n_eval_ep=n_eval_ep, eval_freq=512, n_eval_envs=1, deterministic_eval=True, # selection criteria
            selec_criterion=criterion
            )
        
    def PPO_Ant(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):
        '''
        Ant-v4: 
            normalize: true
            n_timesteps: !!float 1e6
            policy: 'MlpPolicy'
        '''


        print('Sufficient resources: The default execution with PPO and Ant uses only 1 CPUs,')
        print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

        Options.OnPolicy_learn_process(
            'PPO','Ant-v4', # pack
            seed,1e6+.5*1e6,'SB3_PPO_Ant',library_dir,save_policies=False, # learning process
            truth_n_workers=available_resources,normalize=True, # learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy', # learning process parameters
            callback=True, n_eval_ep=n_eval_ep, eval_freq=2048, deterministic_eval=True, # selection criteria
            selec_criterion=criterion
            )

    def PPO_HalfCheetah(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):
        '''
        HalfCheetah-v4:
            normalize: true
            n_envs: 1
            policy: 'MlpPolicy'
            n_timesteps: !!float 1e6
            batch_size: 64
            n_steps: 512
            gamma: 0.98
            learning_rate: 2.0633e-05
            ent_coef: 0.000401762
            clip_range: 0.1
            n_epochs: 20
            gae_lambda: 0.92
            max_grad_norm: 0.8
            vf_coef: 0.58096
            policy_kwargs: "dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activation_fn=nn.ReLU,
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            )"
        '''


        print('Sufficient resources: The default execution with PPO and HalfCheetah uses only 1 CPUs,')
        print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

        Options.OnPolicy_learn_process(
            'PPO','HalfCheetah-v4', # pack
            seed,1e6+.5*1e6,'SB3_PPO_HalfCheetah',library_dir,save_policies=False, # learning process
            truth_n_workers=available_resources, batch_size=64,n_epoch=20,n_steps_per_env=512, normalize=True,# learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gamma=0.98,learning_rate=2.0633e-05,ent_coef=0.000401762,
            clip_range=0.1,gae_lambda=0.92, max_grad_norm=0.8,vf_coef=0.58096,
            policy_kwargs=dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activation_fn=nn.ReLU,
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            ), # learning process parameters
            callback=True, n_eval_ep=n_eval_ep, eval_freq=512, deterministic_eval=True, # selection criteria
            selec_criterion=criterion
            )

    def PPO_Hopper(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):
        '''
        Hopper-v4:
            normalize: true
            n_envs: 1
            policy: 'MlpPolicy'
            n_timesteps: !!float 1e6
            batch_size: 32
            n_steps: 512
            gamma: 0.999
            learning_rate: 9.80828e-05
            ent_coef: 0.00229519
            clip_range: 0.2
            n_epochs: 5
            gae_lambda: 0.99
            max_grad_norm: 0.7
            vf_coef: 0.835671
            policy_kwargs: "dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activnn.ReLU,ation_fn=
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            )"
        '''



        print('Sufficient resources: The default execution with PPO and Hopper uses only 1 CPUs,')
        print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

        Options.OnPolicy_learn_process(
            'PPO','Hopper-v4', # pack
            seed,1e6+.5*1e6,'SB3_PPO_Hopper',library_dir,save_policies=False, normalize=True, # learning process
            truth_n_workers=available_resources, batch_size=32,n_epoch=5,n_steps_per_env=512,# learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gamma=0.999,learning_rate=9.80828e-05,ent_coef=0.00229519,
            clip_range=0.2,gae_lambda=0.99, max_grad_norm=0.7,vf_coef=0.835671,
            policy_kwargs=dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activation_fn=nn.ReLU,
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            ), # learning process parameters

            callback=True, n_eval_ep=n_eval_ep, eval_freq=512, deterministic_eval=True, # selection criteria
            selec_criterion=criterion
            )

    def PPO_Swimmer(seed,library_dir,criterion,available_resources=None,n_eval_ep=500):
        '''
        Swimmer-v4:
            n_timesteps: !!float 1e6
            policy: 'MlpPolicy'
            gamma: 0.9999
            n_envs: 4
            n_steps: 1024
            batch_size: 256
            learning_rate: !!float 6e-4
            gae_lambda: 0.98
        '''


        if available_resources<4:
            print('Insufficient resources: Running PPO and Swimmer in the default configuration requires 4 CPUs')
        else:
            print('Sufficient resources: The default execution with PPO and Swimmer uses only 4 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OnPolicy_learn_process(
                'PPO','Swimmer-v4', # pack
                seed,1e6+.5*1e6,'SB3_PPO_Swimmer',library_dir,save_policies=False, # learning process
                n_workers=4,truth_n_workers=available_resources, batch_size=256,n_steps_per_env=1024,# learning interaction
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy',gamma=0.9999,learning_rate=6e-4,gae_lambda=0.98,

                callback=True, n_eval_ep=n_eval_ep, eval_freq=1024, n_eval_envs=1,deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )


    # Off-policy packs (with SAC)
    #------------------------  Classic Control
    def SAC_Pendulum(seed,library_dir,criterion,available_resources=None,n_eval_ep=5):

        ''' Pendulum-v1:
            n_timesteps: 20000
            policy: 'MlpPolicy'
            learning_rate: !!float 1e-3

        '''


        if available_resources<1:
            print('Insufficient resources: Running SAC and Pendulum in the default configuration requires 1 CPUs')
        else:
            print('Sufficient resources: The default execution with SAC and Pendulum uses only 1 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OffPolicy_learn_process(
                'SAC','Pendulum-v1', # pack
                seed,20000+.5*20000,'SB3_SAC_Pendulum',library_dir,save_policies=False, # learning process
                truth_n_workers=available_resources, # learning interaction
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy', learning_rate=1e-3,# learning process parameters
                # NOTE: si no queremos modificar la configuracion por defecto, tendremos que bajar el valor de n_eval_ep
                # porque sino la ejecucion de prueba va a tardar muchisimo para evaluar todas las politicas de la secuencia
                callback=True, n_eval_ep=n_eval_ep, eval_freq=1, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    def SAC_MountainCarContinuous(seed,library_dir,criterion,available_resources=None,n_eval_ep=5):

        ''' MountainCarContinuous-v0:
            n_timesteps: !!float 50000
            policy: 'MlpPolicy'
            learning_rate: !!float 3e-4
            buffer_size: 50000
            batch_size: 512
            ent_coef: 0.1
            train_freq: 32
            gradient_steps: 32
            gamma: 0.9999
            tau: 0.01
            learning_starts: 0
            use_sde: True
            policy_kwargs: "dict(log_std_init=-3.67, net_arch=[64, 64])"

        '''


        if available_resources<1:
            print('Insufficient resources: Running SAC and MountainCarContinuous in the default configuration requires 1 CPUs')
        else:
            print('Sufficient resources: The default execution with SAC and MountainCarContinuous uses only 1 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OffPolicy_learn_process(
                'SAC','MountainCarContinuous-v0', # pack
                seed,50000+.5*50000,'SB3_SAC_MountainCarContinuous',library_dir,save_policies=False, # learning process
                train_freq=32,batch_size=512,n_epoch=32,truth_n_workers=available_resources, # learning interaction
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy', learning_rate=3e-4,buffer_size=50000,ent_coef=0.1,gamma=0.9999,tau=0.01,
                learning_starts=0, use_sde=True, policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),   # learning process parameters
                # NOTE: si no queremos modificar la configuracion por defecto, tendremos que bajar el valor de n_eval_ep
                # porque sino la ejecucion de prueba va a tardar muchisimo para evaluar todas las politicas de la secuencia
                callback=True, n_eval_ep=n_eval_ep, eval_freq=32, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    #------------------------  Box2D
    def SAC_LunarLanderContinuous(seed,library_dir,criterion,available_resources=None,n_eval_ep=5):

        ''' LunarLanderContinuous-v3:
                n_timesteps: !!float 5e5
                policy: 'MlpPolicy'
                batch_size: 256
                learning_rate: lin_7.3e-4
                buffer_size: 1000000
                ent_coef: 'auto'
                gamma: 0.99
                tau: 0.01
                train_freq: 1
                gradient_steps: 1
                learning_starts: 10000
                policy_kwargs: "dict(net_arch=[400, 300])"

        '''


        if available_resources<1:
            print('Insufficient resources: Running SAC and LunarLanderContinuos in the default configuration requires 1 CPUs')
        else:
            print('Sufficient resources: The default execution with SAC and LunarLanderContinuos uses only 1 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OffPolicy_learn_process(
                'SAC','LunarLanderContinuous-v3', # pack
                seed,5e5+.5*5e5,'SB3_SAC_LunarLanderContinuous',library_dir,save_policies=False, # learning process
                train_freq=1,truth_n_workers=available_resources, # learning interaction
                n_epoch=1,batch_size=256, # policy update
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy',gamma=0.99,ent_coef='auto', learning_rate= lambda progress_remaining: 7.3e-4 * progress_remaining,
                buffer_size=1000000,tau=0.01,learning_starts=10000, 
                policy_kwargs=dict(net_arch=[400, 300]),# learning process parameters
                # NOTE: si no queremos modificar la configuracion por defecto, tendremos que bajar el valor de n_eval_ep
                # porque sino la ejecucion de prueba va a tardar muchisimo para evaluar todas las politicas de la secuencia
                callback=True, n_eval_ep=n_eval_ep, eval_freq=1, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )

    def SAC_BipedalWalker(seed,library_dir,criterion,available_resources=None,n_eval_ep=5):

        ''' BipedalWalker-v3:
                n_timesteps: !!float 5e5
                policy: 'MlpPolicy'
                learning_rate: !!float 7.3e-4
                buffer_size: 300000
                batch_size: 256
                ent_coef: 'auto'
                gamma: 0.98
                tau: 0.02
                train_freq: 64
                gradient_steps: 64
                learning_starts: 10000
                use_sde: True
                policy_kwargs: "dict(log_std_init=-3, net_arch=[400, 300])"

        '''


        if available_resources<1:
            print('Insufficient resources: Running SAC and BipedalWalker in the default configuration requires 1 CPUs')
        else:
            print('Sufficient resources: The default execution with SAC and BipedalWalker uses only 1 CPUs,')
            print('to speed up ground truth validation, the indicated available CPUs are used for ground truth validation.')

            Options.OffPolicy_learn_process(
                'SAC','BipedalWalker-v3', # pack
                seed,5e5+.5*5e5,'SB3_SAC_BipedalWalker',library_dir,save_policies=False, # learning process
                train_freq=64,truth_n_workers=available_resources, # learning interaction
                n_epoch=64,batch_size=256, # policy update
                device='auto', vec_env_type='sequential', # execution type
                policy='MlpPolicy',gamma=0.98,ent_coef='auto', learning_rate= 7.3e-4,
                buffer_size=300000,tau=0.02,learning_starts=10000, use_sde=True, 
                policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),# learning process parameters
                # NOTE: si no queremos modificar la configuracion por defecto, tendremos que bajar el valor de n_eval_ep
                # porque sino la ejecucion de prueba va a tardar muchisimo para evaluar todas las politicas de la secuencia
                callback=True, n_eval_ep=n_eval_ep, eval_freq=64, deterministic_eval=True, # selection criteria
                selec_criterion=criterion
                )
    

class Metrics:

    '''
    en esta clase, dada una df_process generada con un metodo-entorno-semilla-criterio, 
    se generan los datos relacionados con las metricas propuestas.

    quiza no es necesario guardarlas, igual es suficiente con que cada metodo devuelva dichos datos en el return
    '''

    def __init__(self,df_process_path):

        self.df_process=pd.read_csv(df_process_path)
        

    def regions(self,w=0.1):

        truth_last=self.df_process['reward_last'].tolist()

        a,b=None,None
        for i in range(len(truth_last)):

            if truth_last[i]-min(truth_last[:i+1])<=np.std(truth_last[:i+1]) :
                if a==None or i-a<int(len(truth_last)*w): # La condicion tiene que cumplirse con una cierta continuidad (no vale que se vuelva a cumprir mucho despues otra vez)
                    a=i
            if max(truth_last[-i-1:])-truth_last[-1-i]<=np.std(truth_last[-i-1:]) :
                if b==None or b-len(truth_last)+i<int(len(truth_last)*w):
                    b=len(truth_last)-i

        # Para que siempre haya un minimo de datos en los intervalos [0,a], [a,b] y [b,T]
        a=min([max([a,int(w*len(truth_last))]),int(2*len(truth_last)/3)-int(w*len(truth_last))])
        b=max([min([b,len(truth_last)-int(w*len(truth_last))]),a+int(w*len(truth_last))])

        return [a,b]

    def degradation(self):

        truth_last=self.df_process['reward_last'].tolist()

        deg_evol=[]
        for i in range(1,len(truth_last)):
            best_truth=max(truth_last[:i+1])
            last_truth=truth_last[i]
            current_mean_truth=np.mean(truth_last[:i+1])
    
            if best_truth!=last_truth:
                degradation_level=(best_truth-last_truth)/(best_truth-min(current_mean_truth,last_truth))
            else:
                degradation_level=0
    
            deg_evol.append(degradation_level)
        return deg_evol
        
    def accuracy(self):

        # FIXME: cuando el criterio es validation, el truth_best deberia ser al asociado al tiempo t+t', sumando el tiempo extra de validacion
        truth_selec=self.df_process['reward_select'].tolist()
        truth_best=self.df_process['reward_best'].tolist()
        truth_last=self.df_process['reward_last'].tolist()
        truth_min=[min(truth_last[:i+1]) for i in range(len(truth_last))]
        
        acc_evol=[]
        for i in range(1,len(truth_selec)):

            if truth_selec[i]==truth_min[i]:
                acc_evol.append(1)
            else:
                acc_evol.append((truth_selec[i]-truth_min[i])/(truth_best[i]-truth_min[i]))

        return acc_evol

    def efficiency(self):

        truth_last=self.df_process['reward_last'].tolist()
        truth_selec=self.df_process['reward_select'].tolist()

        eff_evol=[]
        for i in range(1,len(truth_last)):
            indice = next(j for j, x in enumerate(truth_last[:i+1]) if x >= truth_selec[i])
            eff_evol.append((indice+1)/len(truth_last[:i+1]))

        return eff_evol

class ProcessAnalysis:
    '''
    En esta clase se añadiran los metodos que permitan a partir de un df_process generar las graficas:
    - learning curve criterion vs truth learning curve
    - evolution of the truth reward of the last policy + region limits
    - boxplots of degradation, accuracy and efficiency by region
    '''

    def __init__(self,df_process_path,figure_path,w=0.1):

        self.df_process=pd.read_csv(df_process_path)
        self.df_process_path=df_process_path
        self.figure_path=figure_path

        metrics=Metrics(self.df_process_path)
        self.region_limits=metrics.regions(w=w)
        self.deg_evol=metrics.degradation()
        self.acc_evol=metrics.accuracy()
        self.eff_evol=metrics.efficiency()

        self.criteria_name="_".join(self.df_process_path.split('/')[-1].split('_')[2:]).replace('.csv','')

    def learning_curve(self):

        truth_selec=self.df_process['reward_select']
        truth_best=self.df_process['reward_best']

        plt.figure(figsize=(4, 2.5))
        plt.subplots_adjust(left=0.19, right=0.98,bottom=0.3,  top=0.92 )

        plt.plot(truth_best, color="black", label="ground truth",linewidth=1)
        plt.plot(truth_selec, color="gold", label="criteria",linewidth=1)
        plt.xlabel('learning iteration')
        plt.ylabel('policy return')

        # Leyenda fuera de la gráfica, debajo
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            frameon=False
        )

        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.savefig(self.figure_path+'/learning_curves_'+self.criteria_name+'.pdf')
        plt.savefig(self.figure_path+'/learning_curves_'+self.criteria_name+'.png')

    def process_regions(self):

        truth_last=self.df_process['reward_last']
        a,b=self.region_limits

        plt.figure(figsize=(4, 2.5))
        plt.subplots_adjust(left=0.19, right=0.98,bottom=0.3,  top=0.92 )

        # Curva azul
        plt.plot(truth_last, color="blue",linewidth=1)

        # Líneas verticales continuas
        plt.axvline(x=a, color="red", linewidth=1, label=r"$a$")
        plt.axvline(x=b, color="black", linewidth=1, label=r"$b$")

        # Leyenda debajo de la gráfica
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            frameon=False
        )

        plt.xlabel('learning iteration')
        plt.ylabel('last policy return')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.savefig(self.figure_path+'/regions.pdf')
        plt.savefig(self.figure_path+'/regions.png')

    def criterion_diagnosis(self):

        a, b = self.region_limits

        deg1, deg2, deg3 = self.deg_evol[:a], self.deg_evol[a:b], self.deg_evol[b:]
        acc1, acc2, acc3 = self.acc_evol[:a], self.acc_evol[a:b], self.acc_evol[b:]
        eff1, eff2, eff3 = self.eff_evol[:a], self.eff_evol[a:b], self.eff_evol[b:]

        data = [deg1, acc1, eff1,deg2, acc2, eff2,deg3, acc3, eff3]

        # Posiciones de los boxplots
        positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.grid(axis="y", linestyle="--", alpha=0.5,linewidth=1)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,

            # Caja (IQR)
            boxprops=dict(facecolor="black",edgecolor="white",alpha=0.8,linewidth=1.5),

            # Mediana
            medianprops=dict(color="red",linewidth=2.5),

            # Bigotes
            whiskerprops=dict(color="black",linewidth=1.2),

            # Topes
            capprops=dict( color="black",linewidth=1.2)
        )

        ax.set_xlim(0,12)

        # Etiquetas de cada boxplot
        ax.set_xticks(positions)
        ax.set_xticklabels([
            r"$\delta$", r"$\alpha$", r"$\varepsilon$",
            r"$\delta$", r"$\alpha$", r"$\varepsilon$",
            r"$\delta$", r"$\alpha$", r"$\varepsilon$"
        ])

        # Nombres de las regiones
        region_centers = [2, 6, 10]
        region_names = ["Initialization", "Learning", "Convergence"]

        for x, name in zip(region_centers, region_names):
            ax.text(x,1.02,name,ha="center",va="bottom",transform=ax.get_xaxis_transform(),fontsize=11)

        # Separadores entre regiones
        ax.axvline(4, color="black", linewidth=1)
        ax.axvline(8, color="black", linewidth=1)

        plt.savefig(self.figure_path + "/diagnostic_" + self.criteria_name + ".pdf")
        plt.savefig(self.figure_path + "/diagnostic_" + self.criteria_name + ".png")

def complete_analysis(df_process_path,figure_path,w=0.1):

    analyzer=ProcessAnalysis(df_process_path,figure_path,w)
    analyzer.learning_curve()
    analyzer.process_regions()
    analyzer.criterion_diagnosis()


# Prueba con algunos procesos aislados variados

# PackOptions.PPO_Pendulum(1,'library/results',{ "name": 'last',"kappa": None,"phi": None,"phi_c": None },available_resources=6)
# PackOptions.PPO_Pendulum(1,'library/results',{ "name": 'train',"kappa": 50,"phi": None,"phi_c": None },available_resources=6)
# PackOptions.PPO_Pendulum(1,'library/results',{ "name": 'validation',"kappa": 5,"phi": 10,"phi_c": None },available_resources=6)
# PackOptions.PPO_Pendulum(1,'library/results',{ "name": 'cost_driven_validation',"kappa": 5,"phi": None,"phi_c": 0.2 },available_resources=6)

# PackOptions.SAC_MountainCarContinuous(5,'library/results',{ "name": 'last',"kappa": None,"phi": None,"phi_c": None },available_resources=6)
# PackOptions.SAC_MountainCarContinuous(2,'library/results',{ "name": 'train',"kappa": 50,"phi": None,"phi_c": None },available_resources=6)
# PackOptions.SAC_MountainCarContinuous(2,'library/results',{ "name": 'validation',"kappa": 5,"phi": 10,"phi_c": None },available_resources=6)
# PackOptions.SAC_MountainCarContinuous(2,'library/results',{ "name": 'cost_driven_validation',"kappa": 5,"phi": None,"phi_c": 0.2 },available_resources=6)


# Obtener graficas de df_process

# complete_analysis('library/results/SB3_PPO_Pendulum/seed1/df_process_last_.csv','library/results/SB3_PPO_Pendulum/seed1')
# complete_analysis('library/results/SB3_PPO_Pendulum/seed1/df_process_train_50.csv','library/results/SB3_PPO_Pendulum/seed1')
# complete_analysis('library/results/SB3_PPO_Pendulum/seed1/df_process_validation_5_10.csv','library/results/SB3_PPO_Pendulum/seed1')
# complete_analysis('library/results/SB3_PPO_Pendulum/seed1/df_process_cost_driven_validation_5_0.2.csv','library/results/SB3_PPO_Pendulum/seed1')

# complete_analysis('library/results/SB3_SAC_MountainCarContinuous/seed2/df_process_last_.csv','library/results/SB3_SAC_MountainCarContinuous/seed2')
# complete_analysis('library/results/SB3_SAC_MountainCarContinuous/seed2/df_process_train_50.csv','library/results/SB3_SAC_MountainCarContinuous/seed2')
# complete_analysis('library/results/SB3_SAC_MountainCarContinuous/seed2/df_process_validation_5_10.csv','library/results/SB3_SAC_MountainCarContinuous/seed2')
# complete_analysis('library/results/SB3_SAC_MountainCarContinuous/seed2/df_process_cost_driven_validation_5_0.2.csv','library/results/SB3_SAC_MountainCarContinuous/seed2')

