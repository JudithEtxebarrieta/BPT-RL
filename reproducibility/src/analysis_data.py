'''
This script generates the data frames required for the analysis from the `df_traj.csv` and `df_val.csv`
files generated when running each process with `generate_process_data.py`, corresponding to each environment-seed pair.

The script contains a class with auxiliary functions, secondary classes used by the main classes, and 
main classes that depend on these secondary classes and are responsible for generating the data required 
for the analysis. The classes in this script are executed from the `generate_analysis_data.py` script.
'''

#==================================================================================================
# Dependencies
#==================================================================================================

import os,sys
import pandas as pd
import json
import bz2
import base64
import os
import numpy as np
from tqdm import tqdm
import numpy as np
from reproducibility.src.process_data import *

#==================================================================================================
# Auxiliary class
#==================================================================================================

class DataConverter:
    '''A class containing auxiliary functions for data transformation.'''

    def compress_decompress_list(my_list,compress=True):
        '''It is used to compress or decompress lists stored in database columns.'''
        if compress:
            # Convert the list to a compact JSON string.
            json_str = json.dumps(my_list)

            # Compress the string.
            compressed_data = bz2.compress(json_str.encode('utf-8'))

            # Convert to base64 to store as text in the database.
            compressed_str = base64.b64encode(compressed_data).decode('utf-8')

            return compressed_str
        else:
            # Read the compressed string from the database.
            compressed_data = base64.b64decode(my_list.encode('utf-8'))

            # Uncompress the string.
            json_str = bz2.decompress(compressed_data).decode('utf-8')

            # Convert the JSON string back to a list.
            my_list = json.loads(json_str)

            return my_list
        
    def concat_traj_seq(traj_seq):
        '''Concatenate trajectories from multiple workers, obtained using different policies in the sequence.'''
        return [np.concatenate(row, axis=0) for row in zip(*np.array(traj_seq))]

    def normalize_list(lst,my_min=None,my_max=None):
        '''Normalize the list provided so that the maximum value in it is 1 and the minimum value is 0.'''
        lst = np.array(lst)

        if my_min==None and my_max==None:
            return (lst - np.min(lst)) / (np.max(lst) - np.min(lst))
        else:
            return (lst - my_min) / (my_max - my_min)
        
    def join_df_analysis_seed(path,seeds,only_test_with_cost_truth=False,only_default_est=False):

        if only_test_with_cost_truth:
            df = pd.concat([pd.read_csv(path+'df_test_truth_with_cost'+str(i)+'.csv') for i in seeds],axis=1)
            df = pd.concat([pd.read_csv(path+"df_test_truth.csv"),df ],axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df.to_csv(path+"df_test_truth.csv", index=False)

            # Delete used files.
            for f in [path+'df_test_truth_with_cost'+str(i)+'.csv' for i in seeds]:
                if os.path.exists(f):
                    os.remove(f)

        elif only_default_est:
            df = pd.concat([pd.read_csv(path+'df_train_default_est'+str(i)+'.csv') for i in seeds],axis=1)
            df = pd.concat([pd.read_csv(path+"df_train_est.csv"),df ],axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df.to_csv(path+"df_train_est.csv", index=False)
            

            df = pd.concat([pd.read_csv(path+'df_test_default_est'+str(i)+'.csv') for i in seeds],axis=1)
            df = pd.concat([pd.read_csv(path+"df_test_est.csv"),df ],axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df.to_csv(path+"df_test_est.csv", index=False)

            # Delete used files.
            for f in [path+'df_test_default_est'+str(i)+'.csv' for i in seeds]:
                if os.path.exists(f):
                    os.remove(f)
            for f in [path+'df_train_default_est'+str(i)+'.csv' for i in seeds]:
                if os.path.exists(f):
                    os.remove(f)

        else:
            list_csv_names=['deg_evolution','learning_regions',
                'df_best_truth','df_last_truth','df_train_truth','df_test_truth',
                'df_last_prec','df_train_prec','df_test_prec',
                'df_test_cost','df_test_n_ep',
                'df_last_eff','df_train_eff','df_test_eff',
                'df_last_est','df_train_est','df_test_est'
                ]


            for csv_name in list_csv_names:
                if csv_name!='learning_regions':
                    dfs = [pd.read_csv(path+csv_name+str(i)+".csv") for i in seeds]
                    for i in range(1,len(dfs)):
                        dfs[i] = dfs[i].drop(columns=['n_policy'], errors='ignore') # Para que la columna 'n_policy' no salga repetida

                    df = pd.concat(dfs, axis=1)
                    df.to_csv(path+csv_name+".csv", index=False)

                else:
                    df = pd.concat([pd.read_csv(path+csv_name+str(i)+".csv") for i in seeds],ignore_index=True)
                    df.to_csv(path+csv_name+".csv", index=False)

                # Delete used files.
                for f in [path+csv_name+str(i)+".csv" for i in seeds]:
                    if os.path.exists(f):
                        os.remove(f)

#==================================================================================================
# Secondary classes
#==================================================================================================

class ProcessEstimator:
    '''
    This class, with its main function `compute_estimates` (which depends on the other two functions,
    `read_create_estimates_csv` and `estimate_from_traj`), allows the estimates associated with each 
    policy in a sequence to be computed from the training or test data stored during the execution 
    of a learning process determined by an environment-seed pair (pack).

    This class uses the df_traj.csv and df_val.csv databases stored for each process (environment-seed pair)
    to generate two additional databases, df_traj_estimates.csv and df_val_estimates.csv, containing the 
    corresponding estimates.

    The estimates are the first main calculation performed by the DataGenerator class.
    '''

    def read_create_estimates_csv(path_csv,seq_size,start_iter=0):
        '''
        Creates (if it does not exist) or reads (if it does exist) a database to store the expected episodic reward estimates for each policy in the sequence.
        '''
        if not os.path.exists(path_csv):
            df_estimates = pd.DataFrame({"n_policy": list(range(start_iter,seq_size))})
            os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            df_estimates.to_csv(path_csv, index=False)
        else:
            df_estimates=pd.read_csv(path_csv)

        return df_estimates
        
    def estimate_from_traj(train_rewards_workers,train_ep_ends_workers,n_timesteps_per_iter,n_traj_ep):

        '''
        This is how the expected episodic reward is estimated by the SB3 library using the training trajectories when running executions in parallel (multiple workers).

        This function computes the estimate for all policies in the sequence.

        * `train_rewards`: matrix of trajectory rewards concatenated in the order in which they were created (one row per worker).
        * `train_ep_ends`: matrix of trajectory dones concatenated in the order in which they were created (one row per worker).
        * `n_timesteps_per_iter`: list containing the number of steps consumed per iteration in parallel (if there is more than one worker, the rollout is taken into account).
        * `n_traj_ep`: number of previous episodes used to compute the average.
        '''

        def ER_seq_single_worker(train_rewards,train_ep_ends,time_steps):
            '''
            Returns a list of the episodic rewards (ER) accumulated along the trajectories, and the times (in steps) at which the episodes are completed.
            '''
            current_train_rewards=train_rewards[:time_steps]
            current_train_ep_ends=train_ep_ends[:time_steps]

            ep_rw=[]
            ep_last=[]
            last_i=0
            current_i=0

            for done in current_train_ep_ends:
                current_i+=1
                if done:
                    ep_rw.append(sum(current_train_rewards[last_i:current_i]))
                    ep_last.append(current_i)
                    last_i=current_i

            return ep_rw, ep_last
        
        ep_rw_policy=[]
        for time_steps in n_timesteps_per_iter:
            ep_rw_workers=[]
            ep_last_workers=[]

            # Store the per-episode reward data and when they were stored for each worker.
            for i in range(len(train_rewards_workers)):
                ep_rw, ep_last=ER_seq_single_worker(train_rewards_workers[i],train_ep_ends_workers[i],time_steps)
                ep_rw_workers+=ep_rw
                ep_last_workers+=ep_last

            # Keep the last n_traj_ep episodes based on the time at which they were stored.
            ep_rw_sorted = [x for _, x in sorted(zip(ep_last_workers, ep_rw_workers))] # Sort ep_rw according to ep_last.

            if len(ep_rw_sorted)<n_traj_ep:
                ep_rw_policy.append(np.mean(ep_rw_sorted) if ep_rw_sorted else -np.inf) # There may be cases where, at the beginning, we do not complete even a single episode.

            else:
                ep_rw_policy.append(np.mean(ep_rw_sorted[-n_traj_ep:]))

        return ep_rw_policy

    def compute_estimates(algo,env,seed,n_ep,train_test_estimate,data_common_path):
        '''
        The expected episodic reward (EER) estimates are computed using both training and test data.

        The estimates are stored in additional databases. This allows as many estimates as needed 
        to be computed for different criteria to be simulated (e.g., different numbers of episodes 
        for the training estimate). Storing this data avoids having to perform the same calculations
        multiple times.

        When the estimates are computed using validation data (`train_test_estimate='test'`), not only
        are the EER estimates calculated, but also the additional time required, in seconds.
        '''
        
        # Read the databases associated with the process of interest.
        current_path=data_common_path+'/'+algo+'_'+env+'/'+algo+'_'+env+'_seed'+str(seed)+'_/process_info'
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_test=pd.read_csv(current_path+'/df_val.csv')
        
        # Compute the estimates that will be used as ground truth using the first 500 validation episodes.
        df_test['ep_rewards']=[ DataConverter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
        df_val_estimates=ProcessEstimator.read_create_estimates_csv(current_path+'/df_val_estimates.csv',df_train.shape[0])
        #print([ np.mean(i[:500]) for i in df_test['ep_rewards'] ]==df_val_estimates['truth'].tolist()) NOTE: After some time has passed, repeating the decompression and averaging operations produces different results.
        
        if 'truth' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth']=[ np.mean(i[:500]) for i in df_test['ep_rewards'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        if 'truth_norm' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth_norm']=[ (i-min(df_val_estimates['truth']))/(max(df_val_estimates['truth'])-min(df_val_estimates['truth'])) for i in df_val_estimates['truth'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)

        # Compute estimates from training data.
        if train_test_estimate=='train':
            df_traj_estimates=ProcessEstimator.read_create_estimates_csv(current_path+'/df_traj_estimates.csv',df_train.shape[0])
            
            # Add the required columns to df_traj_estimates if they have not already been computed.
            df_train['traj_rewards']=[ DataConverter.compress_decompress_list(i,compress=False) for i in df_train['traj_rewards']]
            df_train['traj_ep_end']=[ DataConverter.compress_decompress_list(i,compress=False) for i in df_train['traj_ep_end']]
            rollout=np.array(df_train['traj_rewards'][0]).shape[1]
            
            if str(n_ep)+'_traj_ep' not in df_traj_estimates.columns.tolist():
                df_traj_estimates[str(n_ep)+'_traj_ep']=ProcessEstimator.estimate_from_traj(
                    DataConverter.concat_traj_seq(df_train['traj_rewards']),
                    DataConverter.concat_traj_seq(df_train['traj_ep_end']),
                    list(range(rollout,rollout*(df_train.shape[0]+1),rollout)),n_ep)
                df_traj_estimates.to_csv(current_path+'/df_traj_estimates.csv', index=False)

        # Compute estimates from validation data.
        if train_test_estimate=='test':
            # Add the required columns to df_val_estimates if they have not already been computed.
            if str(n_ep)+'_val_ep' not in df_val_estimates.columns.tolist():
                estimates=[np.mean(i[500:(500+n_ep)]) for i in df_test['ep_rewards']]
                val_times=[]
                for i in range(df_test.shape[0]):
                    df_test_elapsed_val_time=DataConverter.compress_decompress_list(df_test['elapsed_val_time'][i],compress=False)
                    df_test_n_val_ep=DataConverter.compress_decompress_list(df_test['n_val_ep'][i],compress=False)
                    val_time_until_truth=df_test_elapsed_val_time[df_test_n_val_ep.index(500)] 
                    val_time=df_test_elapsed_val_time[df_test_n_val_ep.index(500+n_ep)] 
                    val_times.append(val_time-val_time_until_truth)
                          
                df_val_estimates[str(n_ep)+'_val_ep']=[DataConverter.compress_decompress_list(i) for i in zip(estimates,val_times)]
                df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)

class Selector: 

    '''
    The functions in this class return the index of the selected policy in the sequence according to
    different possible criteria.

    For the class functions to work, ProcessEstimator.compute_estimates must have been executed beforehand,
    as they rely on the databases generated by that process: df_traj_estimates.csv and df_val_estimates.csv.
    '''

    def __init__(self,pack,seed):

        global global_data_common_path
          
        # Read the databases for the process of interest.
        current_path=global_data_common_path+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info'
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_train_estimates=pd.read_csv(current_path+'/df_traj_estimates.csv')
        df_test=pd.read_csv(current_path+'/df_val.csv')
        df_test_estimates=pd.read_csv(current_path+'/df_val_estimates.csv')

        self.df_train=df_train
        self.df_test=df_test
        self.df_train_estimates=df_train_estimates
        self.df_test_estimates=df_test_estimates

    def truth_best_policy(self,elapsed_time):
        policy_id=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time)]['truth'].idxmax()
        return policy_id

    def worst_policy(self,elapsed_time):
        policy_id=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time)]['truth'].idxmin()
        return policy_id

    def last_policy(self,elapsed_time):
        '''Given the elapsed learning time, returns the index of the last policy visited.'''
        last_policy=self.df_train[self.df_train['time_seconds']<=elapsed_time]['n_policy'].max()
        return last_policy

    def best_policy_training(self,elapsed_time,n_traj_ep):

        # List of expected episodic reward (EER) estimates computed from the training data for the sequence of policies visited so far.
        estimated_EER_seq=self.df_train_estimates[(self.df_train['time_seconds']<=elapsed_time)][str(n_traj_ep)+'_traj_ep'].tolist()

        # Index of the policy with the highest mean EER in the training data.
        return estimated_EER_seq.index(max(estimated_EER_seq)) 

    def best_policy_validation(self,elapsed_time,n_val_ep,freq):

        # Validation times at the specified constant frequency.
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indices of the policies associated with those times, their EER estimates, and the additional time required to compute them.
        current_val_policies=[]
        esti_time_seq=[]
        for time in current_val_times:
            policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) ].index.max()

            current_val_policies.append(policy_id) 
            esti_time_seq.append(self.df_test_estimates[self.df_test_estimates['n_policy']==policy_id][str(n_val_ep)+'_val_ep'].values[0])

        esti_time_seq= [DataConverter.compress_decompress_list(i,compress=False) for i in esti_time_seq]

        # Split estimates for additional times.
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)

        # Index of the policy (within the subset of policies associated with frequencies) with the highest mean EER in the validation set.
        indx_subseq=estimated_EER_seq.index(max(estimated_EER_seq))

        # Selected policy and total overtime spent on its selection.
        return current_val_policies[indx_subseq], sum(times_seq)
    
    def best_policy_validation_with_cost(self,elapsed_time,cost_perc,n_ep_or_freq):

        # Decompress the required columns from df_test.
        df_test_ep_rewards=self.df_test['ep_rewards']
        df_test_elapsed_val_times=self.df_test['elapsed_val_time']
        df_test_n_val_ep=self.df_test['n_val_ep']

        # Select only the data associated with the episodes reserved for test.
        df_test_ep_rewards=[ep_rewards[500:] for ep_rewards in df_test_ep_rewards]

        # Interaction times up to the current elapsed_time limit.
        current_times = self.df_train.loc[self.df_train['time_seconds'] <= elapsed_time, 'time_seconds'].tolist()

        # Indices of the policies evaluated, their EER estimates and the additional time taken to calculate them.
        current_val_policies=[]
        esti_time_seq=[]
        time_val_n_ep=0 # Estimated cost of validating the policy with n_ep.
        time_one_ep=0 # To store the maximum value observed so far (a pessimistic estimate based on a single episode).
        total_time_val=0 # The total time spent on validation is accumulated here.
        val_n_ep=[0]*self.df_test.shape[0]# We initialize with 0, because only those policies that must be validated will have an n_ep = n_ep_or_freq.
        for time in current_times:

            if time_val_n_ep==0 or (total_time_val+time_val_n_ep)/time<=cost_perc: # We always validate the first policy.

                policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) ].index.max()
                current_val_policies.append(policy_id) 

                val_time_until_truth=df_test_elapsed_val_times[policy_id][df_test_n_val_ep[policy_id].index(500)] 
                current_times=list(np.array(df_test_elapsed_val_times[policy_id])-val_time_until_truth)[500:]
                times_per_ep=[current_times[0]]+[current_times[i]-current_times[i-1] for i in range(1,len(current_times))]

                estimation=np.mean(df_test_ep_rewards[policy_id][:n_ep_or_freq])
                time_val=sum(times_per_ep[:n_ep_or_freq])
                if max(times_per_ep[:n_ep_or_freq])>time_one_ep:
                    time_one_ep=max(times_per_ep[:n_ep_or_freq])
                time_val_n_ep=time_one_ep*n_ep_or_freq # A pessimistic estimate, assuming that the validation of each episode will cost the maximum amount observed to date.

                esti_time_seq.append([estimation,time_val])
                total_time_val+=time_val

                val_n_ep[policy_id]=n_ep_or_freq
                
        # Split estimates of additional times.
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)

        # Index of the policy (in the sequence of validated policies) with the highest mean EER in validation.
        indx_subseq=estimated_EER_seq.index(max(estimated_EER_seq))


        # Selected policy and total overtime spent on its selection.
        return current_val_policies[indx_subseq], sum(times_seq),val_n_ep

class PointEstimator:

    '''
    Given the environment-seed pair that determines the process, the functions in this class allow us to 
    compute the boundaries of the regions and the point values associated with a single iteration of the 
    proposed metrics: degradation, precision (referred to as accuracy in the paper), and efficiency. 
    The EER estimates obtained using the train and validation criteria for the selected policies can also be computed.

    For the class functions to work, ProcessEstimator.compute_estimates must have been executed beforehand,
    as they rely on the databases generated by that process: df_traj_estimates.csv and df_val_estimates.csv.
    '''

    def __init__(self,pack,seed):

        self.selector=Selector(pack,seed)
        self.df_test=self.selector.df_test
        self.df_test_estimates=self.selector.df_test_estimates
        self.df_train=self.selector.df_train
        self.df_train_estimates=self.selector.df_train_estimates

    # Post-hoc diagnostic metrics.
    def learning_region_limits(self):

        truth_last=self.df_test_estimates['truth'].tolist()

        a,b=None,None
        for i in range(len(truth_last)):

            if truth_last[i]-min(truth_last[:i+1])<=np.std(truth_last[:i+1]) :
                if a==None or i-a<int(len(truth_last)*0.1): # The condition must be met with a certain degree of continuity (it is not sufficient for it to be met again much later).
                    a=i
            if max(truth_last[-i-1:])-truth_last[-1-i]<=np.std(truth_last[-i-1:]) :
                if b==None or b-len(truth_last)+i<int(len(truth_last)*0.1):
                    b=len(truth_last)-i

        # To ensure that there is always a minimum amount of data in the intervals [0, a], [a, b] and [b, T].
        a=min([max([a,int(0.1*len(truth_last))]),int(2*len(truth_last)/3)-int(0.1*len(truth_last))])
        b=max([min([b,len(truth_last)-int(0.1*len(truth_last))]),a+int(0.1*len(truth_last))])

        return a,b
   
    def degradation_level(self,elapsed_time):
        ''' 
        Compute degradation as the normalized worsening relative to the mean improvement,
        identified in the code as 'norm_from_mean_worsening_to_improvement'. 
        '''

        # Indices of the last and best policy at the elapsed time.
        truth_best_idx=self.selector.truth_best_policy(elapsed_time)
        last_idx=self.selector.last_policy(elapsed_time)

        # Local degradation between: best-last and best-current minimum.
        best_truth=self.df_test_estimates.loc[truth_best_idx,'truth']
        last_truth=self.df_test_estimates.loc[last_idx,'truth']
        current_mean_truth=min(self.df_test_estimates.loc[:last_idx,'truth'].mean(),last_truth)

        if best_truth!=last_truth:
            degradation_level=(best_truth-last_truth)/(best_truth-current_mean_truth)
        else:
            degradation_level=0

        return degradation_level
    
    def precision(self,elapsed_time,n_policy,normalized,val_time=0):

        '''
        Measure selection precision (accuracy) as the relative EER of the best policy selected 
        by the criterion with respect to the reward of the truth best policy, referred to in the code
        as 'relative_perc_criteria_best' and defined in the range [0, 1], where 1 indicates good 
        performance and 0 indicates poor performance.
        '''

        last_policy=self.selector.last_policy(elapsed_time)
        last_policy_without_val=self.selector.last_policy(elapsed_time-val_time)
        if not normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) ]['truth'].tolist()
            EER_list_without_val=self.df_test_estimates[(self.df_test['n_policy']<=last_policy_without_val) ]['truth'].tolist()
        if normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) ]['truth_norm'].tolist()

        EER_real_best=max(EER_list)

        EER_criteria_best=EER_list[n_policy]

        if EER_criteria_best==min(EER_list_without_val):
            return 1
        else:
            return (EER_criteria_best-min(EER_list_without_val))/(EER_real_best-min(EER_list_without_val))
 
    def efficiency(self,elapsed_time,n_policy,normalized,metric='first_time_to_same_reward'):

        '''
        Measure selection efficiency as the ratio of the total time required to first reach the same EER,
        identified in the code as 'first_time_to_same_reward'. It is defined in the range [0, 1], where 1
        is good (the entire learning time is required to reach that performance) and 0 is poor (0% of the
        total learning time is required to reach that performance).
        '''
        last_policy=self.selector.last_policy(elapsed_time)

        if not normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) ]['truth'].tolist()
        if normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) ]['truth_norm'].tolist()


        EER_criteria_best=EER_list[n_policy]
        indx_first_time = next((i for i, EER_now in enumerate(EER_list) if EER_now >= EER_criteria_best), None)

        return (indx_first_time+1)/(n_policy+1)
  
    # Estimation of the EER of the selected policy.
    def best_estimation_training(self,elapsed_time,n_traj_ep):

        # List of EER estimates computed from the training data for the sequence of policies visited so far.
        estimated_EER_seq=self.df_train_estimates[(self.df_train['time_seconds']<=elapsed_time)][str(n_traj_ep)+'_traj_ep'].tolist()

        return max(estimated_EER_seq)

    def best_estimation_validation(self,elapsed_time,n_val_ep,freq):

        # Validation times at a specified constant frequency.
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indexes of the policies associated with those periods, their EER estimates and the additional time taken to calculate them.
        current_val_policies=[]
        esti_time_seq=[]
        for time in current_val_times:
            policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) ].index.max()

            current_val_policies.append(policy_id) 
            esti_time_seq.append(self.df_test_estimates[self.df_test_estimates['n_policy']==policy_id][str(n_val_ep)+'_val_ep'].values[0])

        esti_time_seq= [DataConverter.compress_decompress_list(i,compress=False) for i in esti_time_seq]

        # Split estimates for additional times.
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)

        return max(estimated_EER_seq)
    
    def best_estimation_validation_with_cost(self,elapsed_time,cost_perc,n_ep_or_freq,):

        # Extract the required columns from df_test.
        df_test_ep_rewards=[DataConverter.compress_decompress_list(i,compress=False) for i in self.df_test['ep_rewards']]
        df_test_elapsed_val_times=[DataConverter.compress_decompress_list(i,compress=False) for i in self.df_test['elapsed_val_time']]
        df_test_n_val_ep=[DataConverter.compress_decompress_list(i,compress=False) for i in self.df_test['n_val_ep']]

        # Select only the data associated with the episodes reserved for test.
        df_test_ep_rewards=[ep_rewards[500:] for ep_rewards in df_test_ep_rewards]

        # Interaction times up to the current elapsed_time limit.
        current_times = self.df_train.loc[self.df_train['time_seconds'] <= elapsed_time, 'time_seconds'].tolist()

        # Indexes of the policies evaluated, their EER estimates and the additional time taken to calculate them.
        current_val_policies=[]
        esti_time_seq=[]
        time_val_n_ep=0 # Estimated cost of validating the policy with n_ep.
        time_one_ep=0 # To store the maximum value observed so far (a pessimistic estimate based on a single episode).
        total_time_val=0 # The total time spent on validation is being accumulated here.
        val_n_ep=[0]*self.df_test.shape[0] # We initialize with 0, because only those policies that must be validated will have an n_ep = n_ep_or_freq.
        for time in current_times:

            if time_val_n_ep==0 or (total_time_val+time_val_n_ep)/time<=cost_perc: # La primera politica siempre la validamos

                policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) ].index.max()
                current_val_policies.append(policy_id) 

                val_time_until_truth=df_test_elapsed_val_times[policy_id][df_test_n_val_ep[policy_id].index(500)] 
                current_times=list(np.array(df_test_elapsed_val_times[policy_id])-val_time_until_truth)[500:]
                times_per_ep=[current_times[0]]+[current_times[i]-current_times[i-1] for i in range(1,len(current_times))]

                estimation=np.mean(df_test_ep_rewards[policy_id][:n_ep_or_freq])
                time_val=sum(times_per_ep[:n_ep_or_freq])
                if max(times_per_ep[:n_ep_or_freq])>time_one_ep:
                    time_one_ep=max(times_per_ep[:n_ep_or_freq])
                time_val_n_ep=time_one_ep*n_ep_or_freq # Pessimistic estimate, assuming that the validation of each episode will cost the maximum amount observed to date.

                esti_time_seq.append([estimation,time_val])
                total_time_val+=time_val

                val_n_ep[policy_id]=n_ep_or_freq
                
        # Split estimates for additional times.
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)


        # Estimation of the policy selected as the best.
        return max(estimated_EER_seq)


class EvolutionGenerator:
    '''
    This class contains functions that allow us to generate the evolution of the proposed
    post-hoc diagnostic metrics during a process determined by an environment-seed, 
    as well as the evolution of the estimates of the policies selected according to the different criteria.
    '''
    def __init__(self,pack,seed):

        self.estimator=PointEstimator(pack,seed)
        self.estimator.selector.df_test['ep_rewards']=[DataConverter.compress_decompress_list(i,compress=False) for i in self.estimator.selector.df_test['ep_rewards']]
        self.estimator.selector.df_test['elapsed_val_time']=[DataConverter.compress_decompress_list(i,compress=False) for i in self.estimator.selector.df_test['elapsed_val_time']]
        self.estimator.selector.df_test['n_val_ep']=[DataConverter.compress_decompress_list(i,compress=False) for i in self.estimator.selector.df_test['n_val_ep']]

    def degradation_evolution(self,x_times):
        return [self.estimator.degradation_level(time) for time in x_times]

    def precision_evolution(self,x_times,n_ep=None,freq=None,criteria='last',normalized=False,for_analyzer=False,cost_perc=0.1):
        
        y_eff=[]
        x_extras=[]
        val_time=0
        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.selector.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.selector.worst_policy(time)
            if criteria=='last':
                policy_id=self.selector.last_policy(time)
            if criteria=='best_train':
                policy_id=self.selector.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,val_time=self.selector.best_policy_validation(time,n_ep,freq)
                x_extras.append(val_time)
            if criteria =='best_val_with_cost_freq':
                policy_id,val_time,val_n_ep=self.selector.best_policy_validation_with_cost(time,cost_perc,n_ep)
                x_extras.append(val_time)

            if for_analyzer:
                val_time=0


            eff=self.estimator.precision(time+val_time,policy_id,normalized=normalized,val_time=val_time)
            y_eff.append(eff)

        if criteria in ['best_val']:
            return y_eff, x_extras
        if criteria in ['best_val_with_cost_freq']:
            return y_eff, x_extras, val_n_ep 
        if criteria not in ['best_val','best_val_with_cost_freq']:
            return y_eff
    
    def efficiency_evolution(self,x_times,n_ep=None,freq=None,criteria='last',normalized=False,metric='first_time_to_same_reward',cost_perc=0.1):
        
        y_eff=[]
        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.selector.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.selector.worst_policy(time)
            if criteria=='last':
                policy_id=self.selector.last_policy(time)
            if criteria=='best_train':
                policy_id=self.selector.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,_=self.selector.best_policy_validation(time,n_ep,freq)
            if criteria =='best_val_with_cost_freq':
                policy_id,_,_=self.selector.best_policy_validation_with_cost(time,cost_perc,n_ep)

            eff=self.estimator.efficiency(time,policy_id,normalized=normalized,metric=metric)
            y_eff.append(eff)


        return y_eff
    
    def truth_evolution(self,x_times,n_ep=None,freq=None,criteria='last',cost_perc=0.1):

        '''
        Evolution of the truth expected episodic reward of the policies selected according to the specified criterion.

        * `criteria`: can be 'truth_best', 'worst', 'last', 'best_train', 'best_val', or 'best_val_with_cost'.
        '''

        y_truth=[]

        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.selector.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.selector.worst_policy(time)
            if criteria=='last':
                policy_id=self.selector.last_policy(time)
            if criteria=='best_train':
                policy_id=self.selector.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,_=self.selector.best_policy_validation(time,n_ep,freq)
            if criteria =='best_val_with_cost_freq':
                policy_id,_,_=self.selector.best_policy_validation_with_cost(time,cost_perc,n_ep)
            
            y_truth.append(self.selector.df_test_estimates[self.selector.df_test_estimates['n_policy']==policy_id]['truth'].values[0])
        
        return y_truth

    def estimation_evolution(self,x_times,n_ep=None,freq=None,criteria='last',cost_perc=0.1):

        '''
        Evolucion del expected episodic reward estimado de las politicas seleccionadas por el criterio indicado
        
        :param criteria: puede ser 'truth_best', 'worst', 'last', 'best_train', 'best_val'
        '''

        y_estimation=[]

        for time in x_times:
            if criteria=='best_train':
                estimation=self.estimator.best_estimation_training(time,n_ep)
            if criteria=='best_val':
                estimation=self.estimator.best_estimation_validation(time,n_ep,freq)
            if criteria=='best_val_with_cost_freq':
                estimation=self.estimator.best_estimation_validation_with_cost(time,cost_perc,n_ep)
            
            y_estimation.append(estimation)
        
        return y_estimation  

#==================================================================================================
# Main classes
#==================================================================================================

class DataGenerator:
    '''
    The functions in this class allow adding information to databases that accumulate data from different processes 
    (seeds) of the same environment, including:
    * The evolution of: truth values, estimates, degradations, accuracies, efficiencies, or costs;
    * Learning region boundaries.
    The class initialization computes certain data, which is later completed by calling the other class functions from
    the main script `generate_analysis_data.py`, outside the `__init__` method.

    This class can be considered the main class of the current script, since both its initialization and its functions
    depend on the other classes.
    '''

    def __init__(self,pack,seeds,data_common_path,
                list_freq=[20,15,10,5,2,1],list_n_ep=[500,250,100,50,25,5], list_cost_perc=[0.05,0.1,0.15,0.2], 
                global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff'):

        '''
        During initialization, certain truth, cost, and efficiency metrics are not computed. There is no point
        in calculating these values for all the possible criteria generated by the configuration grids, since 
        the subsequent analysis is only performed using the configuration that generally provides the best 
        accuracy results. Therefore, while accuracy must be computed for all values in the configuration grids,
        test truth, test costs, and the efficiency of all criteria are computed outside the __init__ method.
        '''

        global global_data_common_path
        global_data_common_path=data_common_path
        
        data_path=data_common_path+'/'+pack+'/analysis_data/'

        for i in range(len(seeds)):

            # Estimations of train and validation.
            init,algo,env=pack.split('_')
            for n_ep in tqdm(list_n_ep,desc='completing estimates'):
                ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],n_ep,'train',data_common_path)
                ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],n_ep,'test',data_common_path)
            
            # Generals: regions, degradation, ground truth
            self.add_learning_limits_to_csv(data_path+'learning_regions'+str(seeds[i])+'.csv',pack,seeds[i])
            self.add_deg_to_csv(data_path+'deg_evolution'+str(seeds[i])+'.csv',pack,seeds[i],global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
            self.add_truth_to_csv(data_path+'df_best_truth'+str(seeds[i])+'.csv',pack,seeds[i],criteria='truth_best')
            print('Regions, degradations and ground truth completed')

            # Last: truth, accuracy
            self.add_truth_to_csv(data_path+'df_last_truth'+str(seeds[i])+'.csv',pack,seeds[i],criteria='last')
            self.add_criteria_prec_cost_to_csv(data_path+'df_last_prec'+str(seeds[i])+'.csv',pack,seeds[i],criteria='last')
            print('Last: truth and accuracy completed')
            
            # Train: truth, accuracy
            for n_ep in tqdm(list_n_ep,desc='completing train'):
                self.add_truth_to_csv(data_path+'df_train_truth'+str(seeds[i])+'.csv',pack,seeds[i],conf=[n_ep,None],criteria='best_train')
                self.add_criteria_prec_cost_to_csv(data_path+'df_train_prec'+str(seeds[i])+'.csv',pack,seeds[i],conf=[n_ep,None,None],criteria='best_train')
            print('Train: truth and accuracy completed')

            # Validation: truth, accuracy
            for freq in tqdm(list_freq,desc='completing test'):
                for n_ep in list_n_ep:
                    self.add_truth_to_csv(data_path+'df_test_truth'+str(seeds[i])+'.csv',pack,seeds[i],conf=[n_ep,freq],criteria='best_val')
                    self.add_criteria_prec_cost_to_csv([data_path+'df_test_prec'+str(seeds[i])+'.csv',data_path+'df_test_cost'+str(seeds[i])+'.csv'],pack,seeds[i],conf=[n_ep,freq,None],criteria='best_val')
            print('Validation: truth and accuracy completed')

            # Cost-driven validation: accuracy
            for n_ep in tqdm(list_n_ep,desc='completing test with cost para freq'):
                for cost_perc in list_cost_perc:
                    self.add_criteria_prec_cost_to_csv([data_path+'df_test_prec'+str(seeds[i])+'.csv',data_path+'df_test_cost'+str(seeds[i])+'.csv',data_path+'df_test_n_ep'+str(seeds[i])+'.csv'],pack,seeds[i],conf= [n_ep,None,cost_perc],criteria='best_val_with_cost_freq')
            print('Cost-driven validation: accuracy completed')
            
        # Save variables.
        self.data_path=data_path
        self.data_source_path=data_common_path+'/'+pack+'/'
        self.seeds=seeds
        self.pack=pack


    def add_learning_limits_to_csv(self,path,pack,seed):

        '''Get/add learning region boundaries.'''

        # Initialize class that allows the limits to be calculated.
        estimator=PointEstimator(pack,seed)
        a,b= estimator.learning_region_limits()

        # Read/generate a csv file to store the new limits if they have not already been stored.
        if not os.path.exists(path):
            df = pd.DataFrame(columns=['pack_seed','a','b','T'])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
        else:
            df=pd.read_csv(path)

        if not ((df['pack_seed'] == pack+str(seed))).any():
            df.loc[len(df)]=[pack+str(seed),a,b,estimator.df_test.shape[0]]
            df.to_csv(path,index=False)

    def add_deg_to_csv(self,path,pack,seed,global_deg_metric='norm_worsening_to_improvement',local_deg_metric='reward_diff'):
        
        '''Get/add degradation evolution associated with a process.'''

        # Initialize the class that allows calculating evolution of degradation.
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()

        deg=generator.degradation_evolution(x_times)

        # Read/generate a CSV file to store the latest evolution of degradation if it has not already been stored.
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if pack+str(seed)+'_'+global_deg_metric+'_'+local_deg_metric not in df.columns:
            df[pack+str(seed)+'_'+global_deg_metric+'_'+local_deg_metric]=deg
            df.to_csv(path,index=False)

    def add_truth_to_csv(self,path,pack,seed,conf=[None,None],criteria='truth_best',cost_perc=0.1):
        '''
        Adds a column containing the truth values of the policies selected according to the specified criteria, if the column does not already exist.

        * `path`: directory containing the .csv file to which the new truth evolution associated with a process should be added.
        * `criteria`: selection criterion. Possible values are 'truth_best', 'worst', 'last', 'best_train', 'best_val', and 'best_val_with_cost'.
        '''

        # Extracting information from "compressed" variables.
        n_ep,freq=conf

        # Determine the configuration label based on the criteria to be used for the column name.
        if criteria in ['last','truth_best','worst']:
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost_n_ep':
            conf='_'+str(cost_perc)+'cost'+'_'+str(freq)
        if criteria=='best_val_with_cost_freq':
            conf='_'+str(n_ep)+'_'+str(cost_perc)+'cost'

        # Initialize class that allows the calculation of the evolution of truth.
        generator=EvolutionGenerator(pack,seed) 
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Read/generate a csv file to store the new truth data if it has not already been stored.
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if pack+str(seed)+str(conf) not in df.columns:
            truth_evol=generator.truth_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,cost_perc=cost_perc)
            df[pack+str(seed)+str(conf)]=truth_evol
            df.to_csv(path,index=False)
 
             
    def add_criteria_prec_cost_to_csv(self,path,pack,seed,conf=[None,None,None],criteria='truth_best',metric='relative_perc_criteria_best'):
        
        '''
        Adds columns containing the accuracy and cost of the policies selected according to the specified criteria, if they do not already exist.

        * `path`: directory associated with the `.csv` file to which the new accuracy and cost evolution associated with a process should be added. 
        For the `'best_val'` and `'best_val_with_cost'` criteria, `path` has the following structure: `path = [path1, path2]`, where `path1` and 
        `path2` correspond to the accuracy and cost `.csv` files, respectively.
        * `criteria`: selection criterion. Possible values are `'truth_best'`, `'worst'`, `'last'`, `'best_train'`, `'best_val'`, and `'best_val_with_cost'`.
        '''
        
        # Extracting information from "compressed" variables.
        n_ep,freq,cost_perc=conf

        # Determine the configuration label based on the criteria to be used for the column name.
        if criteria=='last':
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost_n_ep':
            conf='_'+str(cost_perc)+'cost_'+str(freq)
        if criteria=='best_val_with_cost_freq':
            conf='_'+str(n_ep)+'_'+str(cost_perc)+'cost'

        # Initialize class that enables the calculation of evolutions.
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Read/generate a csv file to store the new evolution if it has not already been stored.
        if criteria in ['best_val']:
            path_prec,path_cost=path
            df_prec=ProcessEstimator.read_create_estimates_csv(path_prec,generator.estimator.df_train.shape[0])
            df_cost=ProcessEstimator.read_create_estimates_csv(path_cost,generator.estimator.df_train.shape[0])
            if pack+str(seed)+str(conf)+'_'+metric not in df_prec.columns:
                eff_evol,cost_evol=generator.precision_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,metric=metric,for_analyzer=True,cost_perc=cost_perc)
                df_prec[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
                df_cost[pack+str(seed)+str(conf)+'_'+metric]=np.array(cost_evol) / np.array(x_times)
                df_prec.to_csv(path_prec,index=False)
                df_cost.to_csv(path_cost,index=False)

        if criteria in ['best_val_with_cost_n_ep','best_val_with_cost_freq']:
            path_prec,path_cost,path_n_ep=path
            df_prec=ProcessEstimator.read_create_estimates_csv(path_prec,generator.estimator.df_train.shape[0])
            df_cost=ProcessEstimator.read_create_estimates_csv(path_cost,generator.estimator.df_train.shape[0])
            df_n_ep=ProcessEstimator.read_create_estimates_csv(path_n_ep,generator.estimator.df_train.shape[0])
            if pack+str(seed)+str(conf)+'_'+metric not in df_prec.columns:
                eff_evol,cost_evol,val_n_ep=generator.precision_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,metric=metric,for_analyzer=True,cost_perc=cost_perc)
                df_prec[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
                df_cost[pack+str(seed)+str(conf)+'_'+metric]=np.array(cost_evol) / np.array(x_times)
                df_n_ep[pack+str(seed)+str(conf)+'_'+metric]=val_n_ep
                df_prec.to_csv(path_prec,index=False)
                df_cost.to_csv(path_cost,index=False)
                df_n_ep.to_csv(path_n_ep,index=False)

        if criteria in ['last','best_train']: 
            df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
            if pack+str(seed)+str(conf)+'_'+metric not in df.columns:
                eff_evol=generator.precision_evolution(x_times,n_ep=n_ep,freq=None,criteria=criteria,metric=metric,for_analyzer=True,cost_perc=cost_perc)
                df[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
                df.to_csv(path,index=False)

    def add_criteria_eff_to_csv(self,path,pack,seed,conf=[None,None,None],criteria='truth_best',metric='first_time_to_same_reward'):
        
        '''
        Adds a column containing the efficiency of the policies selected according to the specified criteria, if it does not already exist.

        * `path`: directory associated with the `.csv` file to which the new efficiency evolution associated with a process should be added.
        * `criteria`: selection criterion. Possible values are `'truth_best'`, `'worst'`, `'last'`, `'best_train'`, `'best_val'`, and `'best_val_with_cost'`.
       '''
        
        # Extracting information from "compressed" variables.
        n_ep,freq,cost_perc=conf

        # Determine the configuration label based on the criteria to be used for the column name.
        if criteria=='last':
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost_n_ep':
            conf='_'+str(cost_perc)+'cost_'+str(freq)
        if criteria=='best_val_with_cost_freq':
            conf='_'+str(n_ep)+'_'+str(cost_perc)+'cost'

        # Initialize class that enables the calculation of evolutions.
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Read/generate a csv file to store the new evolution if it has not already been stored.
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if pack+str(seed)+str(conf)+'_'+metric not in df.columns:
            eff_evol=generator.efficiency_evolution(x_times,n_ep=n_ep,freq=None,criteria=criteria,metric=metric,cost_perc=cost_perc)
            df[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
            df.to_csv(path,index=False)

    def add_all_criteria_eff(self,optimal_train_n_ep,optimal_test_cost,optimal_test_n_ep,single_seed=None):
        
        if single_seed==None:
            for i in tqdm(range(len(self.seeds))):
                self.add_criteria_eff_to_csv(self.data_path+'df_last_eff.csv',self.pack,self.seeds[i],criteria='last')
                self.add_criteria_eff_to_csv(self.data_path+'df_train_eff.csv',self.pack,self.seeds[i],conf=[optimal_train_n_ep,None,None],criteria='best_train')
                self.add_criteria_eff_to_csv(self.data_path+'df_test_eff.csv',self.pack,self.seeds[i],conf=[optimal_test_n_ep,None,optimal_test_cost],criteria='best_val_with_cost_freq')

        else:
            self.add_criteria_eff_to_csv(self.data_path+'df_last_eff'+str(single_seed)+'.csv',self.pack,single_seed,criteria='last')
            self.add_criteria_eff_to_csv(self.data_path+'df_train_eff'+str(single_seed)+'.csv',self.pack,single_seed,conf=[optimal_train_n_ep,None,None],criteria='best_train')
            self.add_criteria_eff_to_csv(self.data_path+'df_test_eff'+str(single_seed)+'.csv',self.pack,single_seed,conf=[optimal_test_n_ep,None,optimal_test_cost],criteria='best_val_with_cost_freq')

    def add_estimates_to_csv(self,path,pack,seed,conf=[None,None,None],criteria='truth_best'):

        '''
        Adds a column containing the estimates of the policies selected according to the specified criteria, if it does not already exist.

        * `criteria`: selection criterion. Possible values are `'best_train'`, `'best_val'`, and `'best_val_with_cost'`.
       '''

        # Extracting information from "compressed" variables.
        n_ep,freq,cost_perc=conf

        # Determine the configuration label based on the criteria to be used for the column name.
        if criteria=='last':
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost_n_ep':
            conf='_'+str(cost_perc)+'cost_'+str(freq)
        if criteria=='best_val_with_cost_freq':
            conf='_'+str(n_ep)+'_'+str(cost_perc)+'cost'

        # Initialize class that enables the calculation of the evolution of estimates.
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Read/generate a csv file to store the new evolution if it has not already been stored.
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if criteria!='last':
            if pack+str(seed)+str(conf) not in df.columns:
                estimation_evol=generator.estimation_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,cost_perc=cost_perc)
                df[pack+str(seed)+str(conf)]=estimation_evol
                df.to_csv(path,index=False)
        else: # Last is a criterion that does not take estimates into account; therefore, to represent its estimates, we assume that it uses those associated with the most recent policy with the default train.
            if pack+str(seed)+str(conf) not in df.columns:
                current_path=self.data_source_path+pack+'_seed'+str(seed)+'_/process_info/'
                df_train_estimates=pd.read_csv(current_path+'/df_traj_estimates.csv')
                last=np.array(df_train_estimates[conf.replace('_','')+'_traj_ep'].tolist())
                df[pack+str(seed)+str(conf)]=last
                df.to_csv(path,index=False)
         
    def add_all_criteria_estimates(self,default_train_n_ep,default_test_n_ep,default_test_freq,
                                   optimal_train_n_ep,optimal_test_cost,optimal_test_n_ep,single_seed=None):

        if single_seed==None:
            for i in tqdm(range(len(self.seeds))):
                self.add_estimates_to_csv(self.data_path+'df_last_est.csv',self.pack,self.seeds[i],conf=[default_train_n_ep,None,None],criteria='best_train')
                self.add_estimates_to_csv(self.data_path+'df_train_default_est.csv',self.pack,self.seeds[i],conf=[default_train_n_ep,None,None],criteria='best_train')
                self.add_estimates_to_csv(self.data_path+'df_test_default_est.csv',self.pack,self.seeds[i],conf=[default_test_n_ep,default_test_freq,None],criteria='best_val')
                self.add_estimates_to_csv(self.data_path+'df_train_est.csv',self.pack,self.seeds[i],conf=[optimal_train_n_ep,None,None],criteria='best_train')
                self.add_estimates_to_csv(self.data_path+'df_test_est.csv',self.pack,self.seeds[i],conf=[optimal_test_n_ep,None,optimal_test_cost],criteria='best_val_with_cost_freq')

        else:
            self.add_estimates_to_csv(self.data_path+'df_last_est'+str(single_seed)+'.csv',self.pack,single_seed,conf=[default_train_n_ep,None,None],criteria='best_train')
            self.add_estimates_to_csv(self.data_path+'df_train_default_est'+str(single_seed)+'.csv',self.pack,single_seed,conf=[default_train_n_ep,None,None],criteria='best_train')
            self.add_estimates_to_csv(self.data_path+'df_test_default_est'+str(single_seed)+'.csv',self.pack,single_seed,conf=[default_test_n_ep,default_test_freq,None],criteria='best_val')
            self.add_estimates_to_csv(self.data_path+'df_train_est'+str(single_seed)+'.csv',self.pack,single_seed,conf=[optimal_train_n_ep,None,None],criteria='best_train')
            self.add_estimates_to_csv(self.data_path+'df_test_est'+str(single_seed)+'.csv',self.pack,single_seed,conf=[optimal_test_n_ep,None,optimal_test_cost],criteria='best_val_with_cost_freq')


    def add_test_cost_truth(self,optimal_cost,optimal_n_ep,single_seed=None):
        if single_seed==None:
            for i in tqdm(range(len(self.seeds))):
                self.add_truth_to_csv(self.data_path+'df_test_truth.csv',self.pack,self.seeds[i],conf= [optimal_n_ep,None],criteria='best_val_with_cost_freq',cost_perc=optimal_cost)
        else:
            self.add_truth_to_csv(self.data_path+'df_test_truth_with_cost'+str(single_seed)+'.csv',self.pack,single_seed,conf=[optimal_n_ep,None],criteria='best_val_with_cost_freq',cost_perc=optimal_cost)

class PackProcessDataGenerator:

    '''
    The functions in this class are used to generate environment-specific data frames containing estimates,
    estimation costs, and episode sizes across the 30 sequences.

    These functions should be executed after `ProcessEstimator.compute_estimates`, as the 
    `df_analysis_all_seq_est` function specifically depends on the `df_traj_estimates.csv` and `df_val_estimates.csv` databases.

    At the script level, this class can be considered analogous to the `DataGenerator` class,
    as its functions complete the generation of the data required for the analysis using the data collected 
    during the execution of the processes.
    '''

    def df_analysis_all_seq_est(seeds,env,path):

        pack='pack_PPO_'+env
        df_test_all_seq_est = pd.DataFrame()
        df_train_all_seq_est = pd.DataFrame()

        for seed in seeds:  

            path_train=path+'/'+pack.replace('pack','SB3')+'/'+pack+'_seed'+str(seed)+'_/process_info/df_traj_estimates.csv'
            path_test=path+'/'+pack.replace('pack','SB3')+'/'+pack+'_seed'+str(seed)+'_/process_info/df_val_estimates.csv'

            df_train= pd.read_csv(path_train)
            df_test= pd.read_csv(path_test)

            # Select the columns of interest.
            df_train = df_train[[c for c in df_train.columns if 'traj_ep' in c]].copy()
            df_test = df_test[[c for c in df_test.columns if 'val_ep' in c]].copy()

            # Rename.
            df_train = df_train.rename(columns=lambda c: f"{c}_seed{seed}")
            df_test = df_test.rename(columns=lambda c: f"{c}_seed{seed}")

            # Concatenate horizontally.
            df_train_all_seq_est = pd.concat([df_train_all_seq_est, df_train], axis=1)
            df_test_all_seq_est = pd.concat([df_test_all_seq_est, df_test], axis=1)

        # Save complete databases.
        path_train=path+'/'+pack.replace('pack','SB3')+'/analysis_data/df_train_all_seq_est.csv'
        path_test=path+'/'+pack.replace('pack','SB3')+'/analysis_data/df_test_all_seq_est.csv'
        
        df_train_all_seq_est.to_csv(path_train, index=False)
        df_test_all_seq_est.to_csv(path_test, index=False)

    def df_analysis_all_seq_val_cost_len(seeds,env,path):
        pack='pack_PPO_'+env
        path_save=path+'/'+pack.replace('pack','SB3')+'/analysis_data/df_test_all_seq_cost_len.csv'
        val_ep=[500,250,100,50,25,5]

        df_test_all_seq_cost_len = pd.DataFrame()

        for seed in seeds:  

            # columns test: n_policy,ep_inits,ep_rewards,ep_lens,n_val_ep,elapsed_val_time
            # columns train: n_policy,n_timesteps,time_seconds,traj_rewards,traj_ep_end,traj_inits,traj_advantages,traj_values,traj_returns,policy_loss,value_loss,entropy_loss,policy_gradient_loss,KL_div,explained_variance,log_std
            path_test=path+'/'+pack.replace('pack','SB3')+'/'+pack+'_seed'+str(seed)+'_/process_info/df_val.csv'
            path_train=path+'/'+pack.replace('pack','SB3')+'/'+pack+'_seed'+str(seed)+'_/process_info/df_traj.csv'

            df_test=pd.read_csv(path_test)
            df_test['ep_rewards']=[ compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
            df_test['ep_lens']=[ compress_decompress_list(i,compress=False) for i in df_test['ep_lens']]
            df_train=pd.read_csv(path_train)

            df_test_all_seq_cost_len['ep_len_seed'+str(seed)]=[np.mean(i) for i in df_test['ep_lens']]

            for n_ep in val_ep:

                # Validation-to-learn cost ratios
                train_times=df_train['time_seconds'].tolist()
                train_iter_times= [0] + [train_times[i] - train_times[i - 1]  for i in range(1, len(train_times)) ]

                val_times=[]
                for i in range(df_test.shape[0]):
                    df_test_elapsed_val_time=compress_decompress_list(df_test['elapsed_val_time'][i],compress=False)
                    df_test_n_val_ep=compress_decompress_list(df_test['n_val_ep'][i],compress=False)
                    val_time_until_truth=df_test_elapsed_val_time[df_test_n_val_ep.index(500)] 
                    val_time=df_test_elapsed_val_time[df_test_n_val_ep.index(500+n_ep)] 
                    val_times.append(val_time-val_time_until_truth)
                
                val_to_learn_rations=np.array(val_times)/np.array(train_iter_times)
                
                df_test_all_seq_cost_len[str(n_ep)+'_val_ep_seed'+str(seed)]=val_to_learn_rations
                df_test_all_seq_cost_len.to_csv(path_save, index=False)
















