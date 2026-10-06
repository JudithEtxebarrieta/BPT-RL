

#==================================================================================================
# Dependencies
#==================================================================================================

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from final.src.analysis_data import *

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import beta
from matplotlib.lines import Line2D
import math
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

#==================================================================================================
# Common functions
#==================================================================================================

def abbreviate(env_name: str) -> str:
    NAME_TO_ABBR = {
    'LunarLanderContinuous': 'LLC',
    'BipedalWalker': 'BW',
    'Ant': 'A',
    'Hopper': 'H',
    'HalfCheetah': 'HC',
    'Walker2d': 'W2d',
    'Swimmer': 'S'
}
    return NAME_TO_ABBR.get(env_name, env_name)

def from_df_data_to_graph_data(path,pack,
                                which_graph=None,
                                global_deg_metric='norm_from_mean_worsening_to_improvement',
                                local_deg_metric='reward_diff',
                                prec_metric='relative_perc_criteria_best',
                                eff_metric='first_time_to_same_reward',
                                train_conf=None,test_conf=None,
                                n_ep_type=None):

    '''
    Extract the relevant information from the databases of degradation, accuracy or cost by learning regions,
    in the appropriate format to generate the relevant graphs from that data.
    '''
    
    df_limits=pd.read_csv(path[0])
    df_limits = df_limits[
                    df_limits['pack_seed'].str.contains(pack, na=False) 
                ] # Rows from the pack only.

    if which_graph in ['deg_distribution','last_prec']:

        df=pd.read_csv(path[1])
        df = df.filter(like=pack) # Columns from the pack only.

        if which_graph=='deg_distribution':
            df = df.filter(regex=global_deg_metric+'_'+local_deg_metric+"$") # Only columns from the pack with the specified degradation.
        if which_graph=='last_prec':
            df = df.filter(regex='_'+prec_metric+"$") # Only columns from the pack with the specified accuracy.

        # Store degradation data by region.
        initialization,learning,stabilization=[],[],[]

        for pack_seed in df_limits['pack_seed']:
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

            df_pack_seed= df.filter(like=pack_seed+'_').iloc[:, 0].tolist()
            initialization+=df_pack_seed[:a]
            learning+=df_pack_seed[a:b]
            stabilization+=df_pack_seed[b:]

        return initialization,learning,stabilization
    
    if which_graph=='train_conf_prec':

        def get_conf_from_column_name(column_names):
            conf_list=[]
            for column_name in column_names:
                splited=column_name.split('_')
                conf_list.append(splited[3])

            return sorted(list(set(conf_list)), key=int)[::-1]

        df_prec=pd.read_csv(path[1])
        df_prec = df_prec.filter(like=pack) # # Columns from the pack only.
        df_prec = df_prec.filter(regex=prec_metric+"$") # Only columns from the pack with the specified accuracy metric.

        # Store data according to the configuration for each region.
        matrix_conf_initialization, matrix_conf_learning,matrix_conf_stabilization=[],[],[]

        conf_list=get_conf_from_column_name(df_prec.columns)

        for conf in conf_list:
            prec_initialization,prec_learning,prec_stabilization=[],[],[]

            df_prec_conf=df_prec.filter(like='_'+conf+'_')

            for pack_seed in df_limits['pack_seed']:
                a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                df_prec_conf_pack_seed= df_prec_conf.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                prec_initialization+=df_prec_conf_pack_seed[:a]
                prec_learning+=df_prec_conf_pack_seed[a:b]
                prec_stabilization+=df_prec_conf_pack_seed[b:]

            matrix_conf_initialization.append(prec_initialization)
            matrix_conf_learning.append(prec_learning)
            matrix_conf_stabilization.append(prec_stabilization)

        return matrix_conf_initialization,matrix_conf_learning,matrix_conf_stabilization,conf_list

    if which_graph=='test_conf_prec_cost':

        def get_conf_from_column_name(column_names,n_ep_type):
            
            n_ep_list,freq_list=[],[]
            for column_name in column_names:
                splited=column_name.split('_')
                if splited[4] not in ['','relative']: 
                    if n_ep_type=='with_cost_n_ep':
                        n_ep_list.append(splited[3].replace('cost',''))
                        freq_list.append(splited[4])
                    if n_ep_type=='with_cost_freq':
                        n_ep_list.append(splited[3])
                        freq_list.append(splited[4].replace('cost',''))
                    if n_ep_type=='constant':
                        n_ep_list.append(splited[3])
                        freq_list.append(splited[4])


            return sorted(list(set(n_ep_list)), key=float),sorted(list(set(freq_list)), key=float)


        df_prec=pd.read_csv(path[1])
        df_prec = df_prec.filter(like=pack) # Columns from the pack only.
        df_prec = df_prec.filter(regex=prec_metric+"$") # Only columns from the pack with the specified accuracy metric.

        if n_ep_type=='with_cost_n_ep':
            df_prec = df_prec[[col for col in df_prec.columns if 'cost' in col.split('_')[3]]]
        if n_ep_type=='with_cost_freq':
            df_prec = df_prec[[col for col in df_prec.columns if 'cost' in col.split('_')[4]]]
        if n_ep_type=='constant':
            df_prec=df_prec.loc[:, ~df_prec.columns.str.contains("cost")]

        # Store data according to the configuration for each region.
        matrix1,matrix2,matrix3=[],[],[]
        n_ep_list,freq_list=get_conf_from_column_name(df_prec.columns,n_ep_type)
        for n_ep in n_ep_list:
            n_ep_prec_initialization,n_ep_prec_learning,n_ep_prec_stabilization=[],[],[]

            for freq in freq_list:
                freq_prec_initialization,freq_prec_learning,freq_prec_stabilization=[],[],[]

                if n_ep_type=='constant':
                    df_prec_conf=df_prec.filter(like='_'+n_ep+'_'+freq+'_')
                if n_ep_type=='with_cost_n_ep':
                    df_prec_conf=df_prec.filter(like='_'+n_ep+'cost_'+freq+'_')
                if n_ep_type=='with_cost_freq':
                    df_prec_conf=df_prec.filter(like='_'+n_ep+'_'+freq+'cost_')

                for pack_seed in df_limits['pack_seed']:
                    a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                    b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                    df_prec_conf_pack_seed= df_prec_conf.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                    freq_prec_initialization+=df_prec_conf_pack_seed[:a]
                    freq_prec_learning+=df_prec_conf_pack_seed[a:b]
                    freq_prec_stabilization+=df_prec_conf_pack_seed[b:]

                n_ep_prec_initialization.append(freq_prec_initialization)
                n_ep_prec_learning.append(freq_prec_learning)
                n_ep_prec_stabilization.append(freq_prec_stabilization)

            matrix1.append(n_ep_prec_initialization)
            matrix2.append(n_ep_prec_learning)
            matrix3.append(n_ep_prec_stabilization)

        return matrix1,matrix2,matrix3,n_ep_list,freq_list

    if which_graph in ['last_eff','train_eff','test_eff']:
        df=pd.read_csv(path[1])
        df = df.filter(like=pack) # Columns from the pack only.
        df = df.filter(regex='_'+eff_metric+"$") # Only columns from the pack with the specified accuracy.

        # Store degradation data by region.
        initialization,learning,stabilization=[],[],[]

        if which_graph=='last_eff':
            conf=''
        if which_graph=='train_eff':
            conf=str(int(train_conf))+'_'
        if which_graph=='test_eff':
            conf=test_conf+'_'

        for pack_seed in df_limits['pack_seed']:
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

            df_pack_seed= df.filter(like=pack_seed+'_'+conf).iloc[:, 0].tolist()
            initialization+=df_pack_seed[:a]
            learning+=df_pack_seed[a:b]
            stabilization+=df_pack_seed[b:]

        return initialization,learning,stabilization
    
    if which_graph in ['last_prec','train_prec','test_prec']:
        df=pd.read_csv(path[1])
        df = df.filter(like=pack) # Columns from the pack only.
        df = df.filter(regex='_'+prec_metric+"$") # Only columns from the pack with the specified accuracy.

        # Store degradation data by region.
        initialization,learning,stabilization=[],[],[]

        if which_graph=='last_prec':
            conf=''
        if which_graph=='train_prec':
            conf=train_conf
        if which_graph=='test_prec':
            conf=test_conf

        for pack_seed in df_limits['pack_seed']:
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

            df_pack_seed= df.filter(like=pack_seed+'_'+conf+'_').iloc[:, 0].tolist()
            initialization+=df_pack_seed[:a]
            learning+=df_pack_seed[a:b]
            stabilization+=df_pack_seed[b:]

        return initialization,learning,stabilization

    if which_graph=='how_times_best':
        # Keep only the necessary data.
        df_last=pd.read_csv(path[1])
        df_last = df_last.filter(regex=r''+pack) # Only columns associated with the pack.

        df_train=pd.read_csv(path[2])
        df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Only columns associated with the pack and the specified configuration.

        df_test=pd.read_csv(path[3])
        df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Only columns associated with the pack and the specified configuration.

        # Count the number of times each pair of criteria is better by region (excluding ties).
        last_train1,last_train2,last_train3=[0,0],[0,0],[0,0]
        last_test1,last_test2,last_test3=[0,0],[0,0],[0,0]
        train_test1,train_test2,train_test3=[0,0],[0,0],[0,0]
        len_init,len_learning,len_stabilization=0,0,0
        for pack_seed in df_limits['pack_seed']:

            def update_times_best_with_new_seed(truth1,truth2,a,b,old1,old2,old3):
                truth1=np.array(truth1)
                truth2=np.array(truth2)
                olds=(np.array(old1),np.array(old2),np.array(old3))

                news=[np.array([np.sum(truth1[:a]>truth2[:a]), np.sum(truth1[:a]<truth2[:a])]),
                        np.array([np.sum(truth1[a:b]>truth2[a:b]), np.sum(truth1[a:b]<truth2[a:b])]),
                        np.array([np.sum(truth1[b:]>truth2[b:]), np.sum(truth1[b:]<truth2[b:])])
                        ]

                return [old+new for old,new in zip(olds, news)]
            
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

            truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
            truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
            last_train1,last_train2,last_train3 = update_times_best_with_new_seed(truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()

            last_test1,last_test2,last_test3 = update_times_best_with_new_seed(truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

            train_test1,train_test2,train_test3 = update_times_best_with_new_seed(truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

            len_init+=a
            len_learning+=b-a
            len_stabilization+=len(truth_last)-b

        # Converting from a number of times to a percentage.
        matrix1=np.array([last_train1,last_test1,train_test1])/df_limits['a'].sum()
        matrix2=np.array([last_train2,last_test2,train_test2])/(df_limits['b'] - df_limits['a']).sum()
        matrix3=np.array([last_train3,last_test3,train_test3])/(df_limits['T'] - df_limits['b']).sum()

        return matrix1,matrix2,matrix3,len_init,len_learning,len_stabilization

    if which_graph=='in_which_deg_best':

        # Keep only the necessary data.
        df_last=pd.read_csv(path[1])
        df_last = df_last.filter(regex=r''+pack) # Only columns associated with the pack.

        df_train=pd.read_csv(path[2])
        df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Only columns associated with the pack and the specified configuration

        df_test=pd.read_csv(path[3])
        df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Only columns associated with the pack and the specified configuration

        df_deg=pd.read_csv(path[4])
        df_deg = df_deg.filter(like=pack) # Only columns associated with the pack.
        df_deg = df_deg.filter(regex=global_deg_metric+'_'+local_deg_metric+"$") # Only columns from the pack with the specified degradation.

        # Accumulate the degradations in which each of the criteria for each pair is better.
        last_train1,last_train2,last_train3=[[],[]],[[],[]],[[],[]]
        last_test1,last_test2,last_test3=[[],[]],[[],[]],[[],[]]
        train_test1,train_test2,train_test3=[[],[]],[[],[]],[[],[]]
        for pack_seed in df_limits['pack_seed']:

            def update_deg_best_with_new_seed(deg,truth1,truth2,a,b,old1,old2,old3):
                truth1=np.array(truth1)
                truth2=np.array(truth2)
                deg=np.array(deg)
                olds=[old1,old2,old3]

                news=[[deg[:a][truth1[:a] > truth2[:a]],deg[:a][truth1[:a] < truth2[:a]]],
                [deg[a:b][truth1[a:b] > truth2[a:b]],deg[a:b][truth1[a:b] < truth2[a:b]]],
                [deg[b:][truth1[b:] > truth2[b:]],deg[b:][truth1[b:] < truth2[b:]]]
                ]

                return [[list(sub_old) + list(sub_new) for sub_old, sub_new in zip(old_i, new_i)] for old_i, new_i in zip(olds, news)]
            
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])
            deg=df_deg.filter(like=pack_seed).iloc[:, 0].tolist()


            truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
            truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
            last_train1,last_train2,last_train3 = update_deg_best_with_new_seed(deg,truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            last_test1,last_test2,last_test3 = update_deg_best_with_new_seed(deg,truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

            train_test1,train_test2,train_test3 = update_deg_best_with_new_seed(deg,truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

        return [last_train1,last_test1,train_test1], [last_train2,last_test2,train_test2], [last_train3,last_test3,train_test3]

    if which_graph=='with_what_prec_diff_best':
        # Keep only the necessary data.
        df_last=pd.read_csv(path[1])
        df_last = df_last.filter(regex=r''+pack) # Only columns associated with the pack.

        df_train=pd.read_csv(path[2])
        df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Only columns associated with the pack and the specified configuration.

        df_test=pd.read_csv(path[3])
        df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Only columns associated with the pack and the specified configuration.

        df_prec_last=pd.read_csv(path[4])
        df_prec_last = df_prec_last.filter(regex=r''+pack+'.*_'+prec_metric+'$') # Only columns from the pack with the specified accuracy metric.

        df_prec_train=pd.read_csv(path[5])
        df_prec_train = df_prec_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'_'+prec_metric+'$') # Only columns from the pack with the specified accuracy metric.

        df_prec_test=pd.read_csv(path[6])
        df_prec_test = df_prec_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'_'+prec_metric+'$') # Only columns from the pack with the specified accuracy metric.

        # Accumulate the accuracies in which each of the criteria for each pair is better.
        last_train1,last_train2,last_train3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
        last_test1,last_test2,last_test3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
        train_test1,train_test2,train_test3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
        for pack_seed in df_limits['pack_seed']:

            def update_prec_best_with_new_seed(prec1,prec2,truth1,truth2,a,b,old1,old2,old3):
                truth1=np.array(truth1)
                truth2=np.array(truth2)
                prec1=np.array(prec1)
                prec2=np.array(prec2)
                olds=[old1,old2,old3]

                news=[[[prec1[:a][truth1[:a] > truth2[:a]],prec2[:a][truth1[:a] > truth2[:a]]],[prec2[:a][truth1[:a] < truth2[:a]],prec1[:a][truth1[:a] < truth2[:a]]]],
                [[prec1[a:b][truth1[a:b] > truth2[a:b]],prec2[a:b][truth1[a:b] > truth2[a:b]]],[prec2[a:b][truth1[a:b] < truth2[a:b]],prec1[a:b][truth1[a:b] < truth2[a:b]]]],
                [[prec1[b:][truth1[b:] > truth2[b:]],prec2[b:][truth1[b:] > truth2[b:]]],[prec2[b:][truth1[b:] < truth2[b:]],prec1[b:][truth1[b:] < truth2[b:]]]]
                ]

                return [[[list(x) + list(y) 
                    for x,y in zip(a_ij, b_ij)] 
                    for a_ij, b_ij in zip(a_i, b_i)] 
                    for a_i, b_i in zip(olds, news)]
            
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])
            prec_last=df_prec_last.filter(like=pack_seed+'_'+prec_metric).iloc[:, 0].tolist()
            prec_train=df_prec_train.filter(like=pack_seed+'_'+str(train_conf)+'_'+prec_metric).iloc[:, 0].tolist()
            prec_test=df_prec_test.filter(like=pack_seed+'_'+str(test_conf)+'_'+prec_metric).iloc[:, 0].tolist()


            truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
            truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
            last_train1,last_train2,last_train3 = update_prec_best_with_new_seed(prec_last,prec_train,truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            last_test1,last_test2,last_test3 = update_prec_best_with_new_seed(prec_last,prec_test,truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

            train_test1,train_test2,train_test3 = update_prec_best_with_new_seed(prec_train,prec_test,truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

        return [last_train1,last_test1,train_test1], [last_train2,last_test2,train_test2], [last_train3,last_test3,train_test3]

def successive_halving(data_path,pack,criteria='best',conf=[None,None],list_processes=list(range(1,31)),eta=2):
        '''
        Using the database that records the evolution of the truths for the policies selected by
        the criterion in each process, it generates a list showing the evolution of the ‘truths’, 
        but limited to the lengths that simulate how far each process has been executed in accordance 
        with successive halving.
        '''

        # Read the relevant database.
        if criteria=='best':
            df_est=pd.read_csv(data_path+pack.replace('pack','SB3')+'/df_'+criteria+'_truth.csv')
        else:
            df_est=pd.read_csv(data_path+pack.replace('pack','SB3')+'/df_'+criteria+'_est.csv')
        df_truth=pd.read_csv(data_path+pack.replace('pack','SB3')+'/df_'+criteria+'_truth.csv')
        total_policies=df_truth.shape[0]

        # Reduce df to the columns that record the truth values for the specified criteria configurations.
        if criteria=='train':
            df_est = df_est[[col for col in df_est.columns if col == 'n_policy' or col.endswith('_'+str(conf[0]))]]
            df_truth = df_truth[[col for col in df_truth.columns if col == 'n_policy' or col.endswith('_'+str(conf[0]))]]
        if criteria=='test':
            if type(conf[1])==int: # When test_default.
                df_est = df_est[[col for col in df_est.columns if col == 'n_policy' or col.endswith('_'+str(conf[0])+'_'+str(conf[1]))]]
                df_truth = df_truth[[col for col in df_truth.columns if col == 'n_policy' or col.endswith('_'+str(conf[0])+'_'+str(conf[1]))]]
            if type(conf[1])==float: # When test_with_cost.
                df_est = df_est[[col for col in df_est.columns if col == 'n_policy' or col.endswith('_'+str(conf[0])+'_'+str(conf[1])+'cost')]]
                df_truth = df_truth[[col for col in df_truth.columns if col == 'n_policy' or col.endswith('_'+str(conf[0])+'_'+str(conf[1])+'cost')]]


        # Align the last, est and truth columns.
        if criteria=='last':
            df_est.columns = df_est.columns.str.replace('_100', '', regex=False)
                
        # The number of iterations s (of halvings) and the length of each one r are calculated based on the eta parameter.
        def possible_B_values_for_eta(max_B,eta):
            def compute_s(B, eta):
                input_s = 0 
                output_s=None
                while output_s!=input_s:
                    output_s = math.floor(math.log(B / (input_s + 1), eta))
                    input_s+=1
                    if output_s<0:
                        return False
                return True

            list_B=[]
            for B in range(1,max_B+1):
                if compute_s(B,eta):
                    list_B.append(B)
            return list_B
        
        def compute_s(B, eta):
            input_s = 0 
            output_s=None
            while output_s!=input_s:
                output_s = math.floor(math.log(B / (input_s + 1), eta))
                input_s+=1
                if output_s<0:
                    raise ValueError( 'SH cannot be applied with total_policies='+str(total_policies)+' and eta='+str(eta)+' value combination')

            return input_s
        
        total_policies=max(possible_B_values_for_eta(total_policies,eta))
        s=compute_s(total_policies,eta)

        r=total_policies/(s+1)


        # Apply successive halving.
        list_est_per_halving=[]
        list_truth_per_halving=[]
        for i in range(s+1):

            r_i=math.floor(r*(eta**i))


            df_current = df_est.loc[
                                    df_est['n_policy'] <= r_i,
                                    [col for col in df_est.columns.drop('n_policy') if col.split('_')[2] in [pack.split('_')[2] + str(seed) for seed in list_processes]]
                                ]

            df_truth_current = df_truth.loc[
                                    df_truth['n_policy'] <= r_i,
                                    [col for col in df_truth.columns.drop('n_policy') if col.split('_')[2] in [pack.split('_')[2] + str(seed) for seed in list_processes]]
                                ]

            if i<s:

                # Number of discards in this iteration.
                n_next=math.floor(len(list_processes)/eta)

                # The worst processes.
                worst_processes = df_current.iloc[-1].nsmallest(len(list_processes)-n_next).index
                # Record the evolution of est and truth for the processes we have discarded.
                df_worst = df_current[worst_processes]
                df_truth_worst= df_truth_current[worst_processes]
                list_est_per_halving += [df_worst[col].tolist() for col in df_worst.columns]
                list_truth_per_halving+=[df_truth_worst[col].tolist() for col in df_truth_worst.columns]
                # Update the list of seeds that pass to the next iteration.
                list_best_columns = [col for col in df_current.columns if col not in worst_processes]
                list_processes=[int(s.split('_')[2].replace(pack.split('_')[2], '')) for s in list_best_columns]


            else:
                list_est_per_halving += [df_current[col].tolist() for col in df_current.columns]
                list_truth_per_halving += [df_truth_current[col].tolist() for col in df_truth_current.columns]
       

    
        return list_est_per_halving,list_truth_per_halving

#==================================================================================================
# Class for generating figures of the main analysis
#==================================================================================================

# LaTeX-style figures.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\boldmath
'''

class MainGrapher():

    def __init__(self,data_path,figure_path):

        os.makedirs(data_path+'/paper/', exist_ok=True)
        os.makedirs(figure_path+'/analysis/', exist_ok=True)
        os.makedirs(figure_path+'/appendix/', exist_ok=True)

        self.library='SB3'
        self.all_packs=[ 
            # Box2D
            'pack_PPO_LunarLanderContinuous',
            'pack_PPO_BipedalWalker',
            # MuJoCo
            'pack_PPO_Swimmer',
            'pack_PPO_Hopper',
            'pack_PPO_Ant',
            'pack_PPO_HalfCheetah',
            'pack_PPO_Walker2d'        
                            ]
        self.seeds=list(range(1,31))
        self.data_path=data_path
        self.figure_path=figure_path

    def deg_prec_eff(self):

        # Funcion para extraer datos de interes
        data_deg=[[0,0,0] for _ in range(4)]
        data_acc_last,data_acc_train,data_acc_test=[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)]
        data_eff_last,data_eff_train,data_eff_test=[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)]

        def from_metric_data_to_count(data1,data2,data3,count_list):

            m1,m2,m3=np.median(data1),np.median(data2),np.median(data3)

            for i,m in enumerate([m1,m2,m3]):

                if m<0.25:
                    count_list[i][0]+=1
                if m>=0.25 and m<0.75:
                    count_list[i][1]+=1
                if m>=0.75:
                    count_list[i][2]+=1
            
            for i in range(3):
                count_list[3][i]=count_list[0][i]+count_list[1][i]+count_list[2][i]

            return count_list

        # Funciones para plotear datos pack-region
        def plot_pack_degradation(ax,deg_list,pack_name='',title='',not_last_pack=False,main_title='',xlabel=''):

            data = np.array(deg_list)

            bins = 10
            bin_edges = np.linspace(0, 1, bins + 1)
            counts, _ = np.histogram(data, bins=bin_edges)
            
            percentages = counts / counts.sum() 

            # Escala de gris
            colors = [str(1 - p) for p in percentages]

            bin_width = bin_edges[1] - bin_edges[0]

            # Dibujar barras todas de la misma altura (e.g, 1)
            for left, color in zip(bin_edges[:-1], colors):
                ax.bar(left, 1, width=bin_width, color=color, edgecolor=None, align='edge')

            # Linea de la mediana
            ax.axvline(np.median(data), color='red',linewidth=2)

            ax.set_xlim(-0.05, 1.05)
            ax.set_title(title,fontsize=14)
            ax.text( 0.5, 1.5, rf'\textbf{{{main_title}}}',transform=ax.transAxes,ha='center',va='bottom',fontsize=14)
            ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20,fontsize=12)
            ax.yaxis.set_label_coords(-0.15, 0.3) 
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'],fontsize=12)

            if not_last_pack:   
                ax.set_xticks([])
            else:
                ax.set_xlabel(xlabel,fontsize=12)
            ax.set_yticks([]) 

        def plot_pack_criteria_prec_or_eff(ax,listas,pack_name='',title='',last_pack=False,main_title='',xlabel=''):
            colors=['green','orange','blue']
            color_marker_map = {'green': 's','orange': '^','blue': 'o'}
            for y, data,color in zip(range(len(listas)), listas,colors):

                ax.hlines(y, np.percentile(data, 25), np.percentile(data, 75), color=color,linewidth=1)
                ax.vlines(np.percentile(data, 25), y - 0.2, y + 0.2, color=color)
                ax.vlines(np.percentile(data, 75), y - 0.2, y + 0.2, color=color)
                ax.plot(np.median(data), y, color_marker_map[color], color=color,markersize=4)

            ax.grid(axis='x', linestyle='--',alpha=0.4)
            ax.set_title(title,fontsize=14)
            ax.text( 0.5, 1.5, rf'\textbf{{{main_title}}}',transform=ax.transAxes,ha='center',va='bottom',fontsize=14)
            ax.set_xlim(-0.05,1.05)
            ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20,fontsize=12)
            ax.yaxis.set_label_coords(-0.15, 0.3) 
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'])
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=12)
            if not last_pack:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                if xlabel!='':
                    ax.set_xlabel(xlabel,fontsize=12)
                if pack_name!='' and prec_or_eff=='prec':
            
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='blue', label=r'\textit{Last}', markersize=6, linestyle=''),
                        Line2D([0], [0], marker='^', color='orange', label=r'\textit{Train}', markersize=6, linestyle=''),
                        Line2D([0], [0], marker='s', color='green', label=r'\textit{Validation}', markersize=6, linestyle='')
                    ]
                    ax.legend(handles=legend_elements, loc='upper center',bbox_to_anchor=(1.5, -0.7), ncol=3, frameon=False,fontsize=14)
            
        # Cargar datos necesarios
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'
            
        # Graficas
        fig, axs = plt.subplots(len(self.all_packs),3*3+2, figsize=(5*3,0.65*len(self.all_packs)),
                                    gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1,0.35,1,1,1]})
        plt.subplots_adjust(top=0.9,bottom=0.15,left=0.03,right=0.99, hspace=0.0,wspace=0.0)

        #---- Degradacion
        for i,pack in enumerate(self.all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            # Obtener los datos de degradacion por region de este pack
            deg1,deg2,deg3=from_df_data_to_graph_data([pack_path+'learning_regions.csv',pack_path+'deg_evolution.csv'],pack,'deg_distribution') 

            # Dibujar distribucion de degradacion de las tres regiones
            title=[['Initialization','Learning','Convergence']if i==0 else ['']*3][0]
            main_title=[r'\textbf{Degradation}' if i==0 else ''][0]
            xlabel=r'$\delta$'
            not_last_pack=[False if i==len(self.all_packs)-1 else True][0]
            plot_pack_degradation(axs[i,0],deg1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,1],deg2,title=title[1],not_last_pack=not_last_pack,main_title=main_title,xlabel=xlabel)
            plot_pack_degradation(axs[i,2],deg3,title=title[2],not_last_pack=not_last_pack)

            # Actualizar datos de interes
            data_deg=from_metric_data_to_count(deg1,deg2,deg3,data_deg)

        #---- Precision
        prec_or_eff='prec'
        for i,pack in enumerate(self.all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            # Obtener los datos de prec/eff por region de este pack
            last1,last2,last3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_'+prec_or_eff+'.csv'],pack,which_graph='last_'+prec_or_eff)
            train1,train2,train3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_'+prec_or_eff+'.csv'],pack,which_graph='train_'+prec_or_eff,
                            train_conf=train_conf
                        )
            test1,test2,test3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_'+prec_or_eff+'.csv'],pack,which_graph='test_'+prec_or_eff,
                            test_conf=test_conf
                        )
            
            # Dibujar distribucion de prec/eff de las tres regiones
            title=[['Initialization','Learning ','Convergence']if i==0 else ['']*3][0]
            main_title=[r'\textbf{Accuracy}' if i==0 else ''][0]
            xlabel=r'$\alpha$'
            last_pack=[True if i==len(self.all_packs)-1 else False][0]
            plot_pack_criteria_prec_or_eff(axs[i,4],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,5],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack,main_title=main_title,xlabel=xlabel)
            plot_pack_criteria_prec_or_eff(axs[i,6],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)

            # Actualizar datos de interes
            data_acc_last=from_metric_data_to_count(last1,last2,last3,data_acc_last)
            data_acc_train=from_metric_data_to_count(train1,train2,train3,data_acc_train)
            data_acc_test=from_metric_data_to_count(test1,test2,test3,data_acc_test)

        #---- Eficacia
        prec_or_eff='eff'
        for i,pack in enumerate(self.all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            # Obtener los datos de prec/eff por region de este pack
            last1,last2,last3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_'+prec_or_eff+'.csv'],pack,which_graph='last_'+prec_or_eff,
                            )
            train1,train2,train3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_'+prec_or_eff+'.csv'],pack,which_graph='train_'+prec_or_eff,
                            train_conf=train_conf
                        )
            test1,test2,test3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_'+prec_or_eff+'.csv'],pack,which_graph='test_'+prec_or_eff,
                            test_conf=test_conf
                        )

            # Dibujar distribucion de prec/eff de las tres regiones
            title=[['Initialization','Learning','Convergence']if i==0 else ['']*3][0]
            main_title=[r'\textbf{Efficiency}' if i==0 else ''][0]
            xlabel=r'$\varepsilon$'
            last_pack=[True if i==len(self.all_packs)-1 else False][0]
            plot_pack_criteria_prec_or_eff(axs[i,8],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,9],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack,main_title=main_title,xlabel=xlabel)
            plot_pack_criteria_prec_or_eff(axs[i,10],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)

            # Actualizar datos de interes
            data_eff_last=from_metric_data_to_count(last1,last2,last3,data_eff_last)
            data_eff_train=from_metric_data_to_count(train1,train2,train3,data_eff_train)
            data_eff_test=from_metric_data_to_count(test1,test2,test3,data_eff_test)


        for i in range(len(self.all_packs)):
                axs[i,3].axis('off')
                axs[i,7].axis('off')

        # Guardar base de datos que resume de manera aseptica lo que sucede en la figura enumerando casos
        all_data=[data_deg,data_acc_last,data_acc_train,data_acc_test,
                  data_eff_last,data_eff_train,data_eff_test]
        metric_criteria=[['deg',None],['acc','last'],['acc','train'],['acc','test'],
                         ['eff','last'],['eff','train'],['eff','test']]
        
        df = pd.DataFrame(columns=['Metric', 'Criterion', 'Low', 'Middle','High','Total'])

        cases=len(self.all_packs)
        for case, data in zip(metric_criteria,all_data):
            df.loc[len(df)] = case+ [i for i in data[-1]]+[cases*3]
        
        cases=len(self.all_packs)*3
        data=np.array([j for i in data_acc_last for j in i])+np.array([j for i in data_acc_train for j in i])+np.array([j for i in data_acc_test for j in i])
        df.loc[len(df)] = ['acc','all']+[i for i in data[-3:]]+[cases*3]
        data=np.array([j for i in data_eff_last for j in i])+np.array([j for i in data_eff_train for j in i])+np.array([j for i in data_eff_test for j in i])
        df.loc[len(df)] = ['eff','all']+[i for i in data[-3:]]+[cases*3]


        df.to_latex(self.data_path+'/paper/df_deg_acc_eff_all.tex',
                index=False,
                escape=False,
                float_format="%.2f",
                column_format="|c|c|c|c|c|"
            )

        plt.savefig(self.figure_path+'/analysis/degradation_precision_efficiency.pdf')
 
    def patterns_deg_acc_eff(self):

        # Plotear scatter plot para pack-region
        def plot_uncertainty_rectangles(ax, lista1, lista2, lista3, lista4,color,pack='',first_row=True,first_column=True,center_row=True):

            x_med = np.median(lista1)
            y_lists = [lista2, lista3, lista4]
            markers = ['o', '^', 's'] 

            for y_data, marker in zip(y_lists, markers):
                y_med = np.median(y_data)
                ax.scatter(x_med,y_med,facecolors='none',edgecolor=color,marker=marker,s=40,zorder=3)

                ax.set_ylim(-0.15,1.15)
                ax.set_xlim(-0.1,1.1)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                ax.axvline(0.25, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)
                ax.axvline(0.75, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)
                ax.axhline(0.25, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)
                ax.axhline(0.75, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)

                
                if first_row:
                    ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=9)
                    ax.set_xticks([])
                else:
                    if center_row: 
                        ax.set_xlabel(r'$\delta$')
                if not first_column:
                    ax.set_yticks([])
                else:
                    if first_row:
                        ax.set_ylabel(r'$\alpha$')
                    else:
                        ax.set_ylabel(r'$\varepsilon$')

        
        # Cargar datos necesarios
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'
            
        # Graficas
        fig, axs = plt.subplots(2,len(self.all_packs), figsize=(1.3*len(self.all_packs),1.2*2))
        plt.subplots_adjust(top=0.9,bottom=0.16,left=0.05,right=0.8, hspace=0,wspace=0)

        
        for i,pack in enumerate(self.all_packs):
            first_column=[True if i==0 else False][0]
            center_row=[True if i==len(self.all_packs)//2 else False][0]
            colors = ["#693106", "#BB8A20", "#bba86a"][::-1]
            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            #---- Degradacion
            deg1,deg2,deg3=from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'deg_evolution.csv'],pack,'deg_distribution'
                                ) 

            #---- Precision
            last1,last2,last3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_prec.csv'],pack,which_graph='last_prec'
                        )
            train1,train2,train3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_prec.csv'],pack,which_graph='train_prec',
                            train_conf=train_conf
                        )
            test1,test2,test3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_prec.csv'],pack,which_graph='test_prec',
                            test_conf=test_conf
                        )
            
            plot_uncertainty_rectangles(axs[0,i],deg1,last1,train1,test1,colors[0],pack,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[0,i],deg2,last2,train2,test2,colors[1],pack,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[0,i],deg3,last3,train3,test3,colors[2],pack,first_column=first_column,center_row=center_row)
            
            #---- Eficacia
            last1,last2,last3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_eff.csv'],pack,which_graph='last_eff',
                        )
            train1,train2,train3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_eff.csv'],pack,which_graph='train_eff',
                            train_conf=train_conf
                        )
            test1,test2,test3=from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_eff.csv'],pack,which_graph='test_eff',
                            test_conf=test_conf
                        )
            
            plot_uncertainty_rectangles(axs[1,i],deg1,last1,train1,test1,colors[0],pack,False,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[1,i],deg2,last2,train2,test2,colors[1],first_row=False,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[1,i],deg3,last3,train3,test3,colors[2],first_row=False,first_column=first_column,center_row=center_row)

        # Leyenda

        legend_elements = [
            # puntos (solo borde)
            Line2D([0], [0], marker='o', linestyle='None',
                markerfacecolor='none', markeredgecolor='black',
                label=r'\textit{Last}'),

            Line2D([0], [0], marker='^', linestyle='None',
                markerfacecolor='none', markeredgecolor='black',
                label=r'\textit{Train}'),

            Line2D([0], [0], marker='s', linestyle='None',
                markerfacecolor='none', markeredgecolor='black',
                label=r'\textit{Validation}'),

            # líneas
            Line2D([0], [0], color=colors[0], lw=2,
                label='Initialization'),

            Line2D([0], [0], color=colors[1], lw=2,
                label='Learning'),

            Line2D([0], [0], color=colors[2], lw=2,
                label='Convergence'),
        ]

        axs[0,6].legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(2, 1),  # fuera abajo
            ncol=1,                       # 3 columnas
            frameon=False,
            fontsize=10,
            handlelength=2
        )

            
        plt.savefig(self.figure_path+'/analysis/patterns_degradation_precision_efficiency.pdf')
 
    def criteria_comparison(self,comparison_type='how_times_best'):
        '''
        `comparison_type`: puede ser 'how_times_best', 'in_which_deg_best'
        '''

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        legend_map = {
                "last":  Line2D([0], [0], marker='o', color='#0000FF', label=r'\textit{Last}', markersize=6, linestyle=''),
                "train": Line2D([0], [0], marker='^', color='#FF8800', label=r'\textit{Train}', markersize=6, linestyle=''),
                "test":  Line2D([0], [0], marker='s', color='green',   label=r'\textit{Validation}', markersize=6, linestyle='')
            }

        legend_groups = [["last", "train"],["last", "test"],["train", "test"]]

        # Funcion para guardar datos de interes
        if comparison_type=='how_times_best':
            last_train,last_test,train_test=[[0,0,0] for _ in range(3)],[[0,0,0] for _ in range(3)],[[0,0,0] for _ in range(3)]
        if comparison_type=='in_which_deg_best':
            last_train,last_test,train_test=[[[0,0,0] for _ in range(4)] for _ in range(2)],[[[0,0,0] for _ in range(4)] for _ in range(2)],[[[0,0,0] for _ in range(4)] for _ in range(2)]

        def from_data_times_to_mean_perc(data1,data2,data3,last_train,last_test,train_test):
            
            for i,data in enumerate([data1,data2,data3]):


                last_train[i][0]+=data[0][0]
                last_train[i][2]+=data[0][1]
                last_train[i][1]+=1-data[0][0]-data[0][1]

                last_test[i][0]+=data[1][0]
                last_test[i][2]+=data[1][1]
                last_test[i][1]+=1-data[1][0]-data[1][1]

                train_test[i][0]+=data[2][0]
                train_test[i][2]+=data[2][1]
                train_test[i][1]+=1-data[2][0]-data[2][1]


            return last_train,last_test,train_test

        def from_data_conditioned_deg_to_perc(data1,data2,data3,last_train,last_test,train_test):

            for i,deg_pair in enumerate([data1,data2,data3]):

                #<0.25
                last_train[0][i][0]+= sum(d < 0.25 for d in deg_pair[0][0])
                last_train[1][i][0]+= sum(d < 0.25 for d in deg_pair[0][1])
                last_test[0][i][0]+= sum(d < 0.25 for d in deg_pair[1][0])
                last_test[1][i][0]+= sum(d < 0.25 for d in deg_pair[1][1])
                train_test[0][i][0]+= sum(d < 0.25 for d in deg_pair[2][0])
                train_test[1][i][0]+= sum(d < 0.25 for d in deg_pair[2][1])

                #>=0.25 y <0.75
                last_train[0][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[0][0])
                last_train[1][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[0][1])
                last_test[0][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[1][0])
                last_test[1][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[1][1])
                train_test[0][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[2][0])
                train_test[1][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[2][1])

                #>=0.75
                last_train[0][i][2]+= sum(d >=0.75 for d in deg_pair[0][0])
                last_train[1][i][2]+= sum(d >=0.75 for d in deg_pair[0][1])
                last_test[0][i][2]+= sum(d >=0.75 for d in deg_pair[1][0])
                last_test[1][i][2]+= sum(d >=0.75 for d in deg_pair[1][1])
                train_test[0][i][2]+= sum(d >=0.75 for d in deg_pair[2][0])
                train_test[1][i][2]+= sum(d >=0.75 for d in deg_pair[2][1])

            for i in range(3):

                last_train[0][3][i]=last_train[0][0][i]+last_train[0][1][i]+last_train[0][2][i]
                last_train[1][3][i]=last_train[1][0][i]+last_train[1][1][i]+last_train[1][2][i]
                last_test[0][3][i]=last_test[0][0][i]+last_test[0][1][i]+last_test[0][2][i]
                last_test[1][3][i]=last_test[1][0][i]+last_test[1][1][i]+last_test[1][2][i]
                train_test[0][3][i]=train_test[0][0][i]+train_test[0][1][i]+train_test[0][2][i]
                train_test[1][3][i]=train_test[1][0][i]+train_test[1][1][i]+train_test[1][2][i]

            return last_train,last_test,train_test

        # Funciones para plotear datos por pack-region
        def plot_how_many_times_better(axs, data, pack_name='', title='', not_last_pack=True):

            color_marker_map = {'green': 's','orange': '^','blue': 'o'}
            colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]

            # recorrer cada región → cada ax
            for i, (ax, (pair, (color_left, color_right))) in enumerate(zip(axs, zip(data, colors))):

                left_val, right_val = pair

                # barras enfrentadas
                ax.barh(
                    y=0,
                    width=left_val,
                    left=0,
                    color=color_left,
                    height=0.8,alpha=0.9
                )

                ax.barh(
                    y=0,
                    width=right_val,
                    left=1 - right_val,
                    color=color_right,
                    height=0.8,alpha=0.9
                )

                for x in [0.25, 0.5, 0.75]:
                    ax.axvline(x, color='black', linestyle='--', linewidth=0.5)

                # marcadores extremos
                marker_left = color_marker_map.get(color_left, 'o')
                marker_right = color_marker_map.get(color_right, 'o')

                if left_val > 0:
                    ax.plot(0, 0, marker=marker_left, color=color_left, markersize=5,markeredgecolor='black',)

                if right_val > 0:
                    ax.plot(1, 0, marker=marker_right, color=color_right, markersize=5,markeredgecolor='black')

                # estética por ax
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
                

                #if not_last_pack:
                ax.set_xticks([])

            axs[0].set_title(title)
            # leyenda solo en último
            if not not_last_pack and pack_name != '':
                for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(handles=handles,loc='upper center',bbox_to_anchor=(1.5, 0.7),ncol=len(handles),frameon=False,fontsize=11,handletextpad=0.4)
                        

            # etiqueta global (opcional)
            for i in range(3):
                axs[i].set_ylabel(abbreviate(pack_name),rotation=0, labelpad=20, fontsize=8)
                axs[i].yaxis.set_label_coords(-0.15, 0.1)

        def plot_in_which_deg_best(axs, data, pack_name='', title='', not_last_pack=True,xlabel=''):

            colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]
            color_marker_map = { 'blue': 'o','orange': '^','green': 's'}

            bins = 10
            hist_height = 0.4

            # recorrer 3 regiones → 3 axes
            for i, (ax, pair, pair_colors) in enumerate(zip(axs, data, colors)):

                bar_height = hist_height / len(pair_colors)

                # NUEVO: calcular histogramas PRIMERO (sin normalizar)
                hist_list = []
                for sublist in pair:
                    sublist = np.asarray(sublist)
                    hist, bin_edges = np.histogram(
                        sublist, bins=bins, range=(0, 1), density=False
                    )
                    hist_list.append(hist)

                # NUEVO: normalización conjunta dentro del pair
                pair_max = max([h.max() if len(h) > 0 else 0 for h in hist_list])
                pair_max = pair_max if pair_max > 0 else 1
                norm_hist_list = [h / pair_max for h in hist_list]
                bin_width = bin_edges[1] - bin_edges[0]


                # Grafica
                for j, (hist, color) in enumerate(zip(norm_hist_list, pair_colors)):

                    # degradado consistente entre las dos sublistas del pair
                    base_rgb = np.array(mcolors.to_rgb(color))
                    colors_scaled = [tuple(base_rgb * p + (1 - p)) for p in hist]
                    bottom = j * bar_height

                    for left, col in zip(bin_edges[:-1], colors_scaled):
                        ax.bar(
                            left,
                            bar_height,
                            width=bin_width,
                            bottom=bottom,
                            align='edge',
                            color=col,
                            edgecolor=None
                        )

                    # mediana
                    if len(pair[j]) > 0:
                        median_val = np.median(pair[j])
                        mid_y = bottom + bar_height / 2
                        marker = color_marker_map.get(color, 'o')

                        ax.plot(
                            median_val,
                            mid_y,
                            marker=marker,
                            color='black',
                            markersize=3
                        )

                # estética por ax
                ax.set_xlim(-0.05, 1.05)
                ax.set_yticks([])
                ax.set_title(title,fontsize=14)
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20,fontsize=12)
                ax.yaxis.set_label_coords(-0.15, 0.3) 
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'],fontsize=12)
                ax.yaxis.set_label_coords(-0.15, 0.3) 

                if not_last_pack:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(xlabel,fontsize=14)

            # leyenda
            if not_last_pack:
                for ax in axs:
                    ax.set_xticklabels([])
            else:
                if pack_name != '':
                    for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(
                            handles=handles,
                            loc='upper center',
                            bbox_to_anchor=(1.5, -0.65),
                            ncol=len(handles),
                            frameon=False,
                            fontsize=14
                        )

        # Graficas
        if comparison_type=='how_times_best':
            # NOTE: para sugerencia de Etor sustituit los [2] por [0.5]
            fig, axs = plt.subplots(len(self.all_packs)*3+2,3, figsize=(5,4.8*0.1*len(self.all_packs)),
                                        gridspec_kw={'height_ratios': [1]*len(self.all_packs)+[2]+[1]*len(self.all_packs)+[2]+[1]*len(self.all_packs)})
            plt.subplots_adjust(top=0.92,bottom=0.05,left=0.1,right=0.98, hspace=0,wspace=0.0)
            #plt.subplots_adjust(top=0.92,bottom=0.1,left=0.1,right=0.98, hspace=0.0,wspace=0.0) # NOTE: para sugerencia de Etor descomentar
        else:
            fig, axs = plt.subplots(len(self.all_packs),3*3+2, figsize=(5*3,0.6*len(self.all_packs)),
                                        gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1,0.35,1,1,1]})
            plt.subplots_adjust(top=0.94,bottom=0.15,left=0.04,right=0.99, hspace=0.0,wspace=0.0)

        for i,pack in enumerate(self.all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'
            title=[['Initialization','Learning','Convergence']if i==0 else ['']*3][0]
            not_last_pack=[False if i==len(self.all_packs)-1 else True][0]

            p=len(self.all_packs)
            
            if comparison_type=='how_times_best':

                # Obtener los datos de la medida de comapracion por region de este pack            
                matrix1,matrix2,matrix3,_,_,_=from_df_data_to_graph_data(
                                        [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                        pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv'],
                                        pack,which_graph=comparison_type,train_conf=train_conf,test_conf=test_conf)
                # Dibujar distribucion de la medida de comparacion de las tres regiones
                plot_how_many_times_better([axs[i,0],axs[p+i+1,0],axs[p*2+i+2,0]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_how_many_times_better([axs[i,1],axs[p+i+1,1],axs[p*2+i+2,1]],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_how_many_times_better([axs[i,2],axs[p+i+1,2],axs[p*2+i+2,2]],matrix3,title=title[2],not_last_pack=not_last_pack)

                # Datos de interes
                last_train,last_test,train_test=from_data_times_to_mean_perc(matrix1,matrix2,matrix3,last_train,last_test,train_test)


            if comparison_type=='in_which_deg_best':
                matrix1,matrix2,matrix3=from_df_data_to_graph_data(
                                [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv',pack_path+'deg_evolution.csv'],
                                pack,which_graph=comparison_type,
                                train_conf=train_conf,test_conf=test_conf
                                )
                xlabel=r'$\delta$'
                plot_in_which_deg_best([axs[i,0],axs[i,4],axs[i,8]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_in_which_deg_best([axs[i,1],axs[i,5],axs[i,9]],matrix2,title=title[1],not_last_pack=not_last_pack,xlabel=xlabel)
                plot_in_which_deg_best([axs[i,2],axs[i,6],axs[i,10]],matrix3,title=title[2],not_last_pack=not_last_pack)

                #Datos de interes
                last_train,last_test,train_test=from_data_conditioned_deg_to_perc(matrix1,matrix2,matrix3,last_train,last_test,train_test)

            if not comparison_type=='how_times_best':
                for i in range(len(self.all_packs)):
                    axs[i,3].axis('off')
                    axs[i,7].axis('off')
            else:
                for i in range(3):
                    axs[p,i].axis('off')
                    axs[p*2+1,i].axis('off')

        # Guardar en df datos de interes que resumen de manera aseptica los numeros de casos de interes que se dan en la figura
        if comparison_type=='how_times_best':        
            df = pd.DataFrame(columns=['case', 'mean_perc_init', 'mean_perc_learn','mean_perc_conv','mean_perc_all'])
            
            df.loc[len(df)] =['last wins train']+list(np.array([last_train[0][0],last_train[1][0],last_train[2][0]])/len(self.all_packs))+[sum([last_train[0][0],last_train[1][0],last_train[2][0]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['train wins last']+list(np.array([last_train[0][2],last_train[1][2],last_train[2][2]])/len(self.all_packs))+[sum([last_train[0][2],last_train[1][2],last_train[2][2]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['last wins test']+list(np.array([last_test[0][0],last_test[1][0],last_test[2][0]])/len(self.all_packs))+[sum([last_test[0][0],last_test[1][0],last_test[2][0]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['test wins last']+list(np.array([last_test[0][2],last_test[1][2],last_test[2][2]])/len(self.all_packs))+[sum([last_test[0][2],last_test[1][2],last_test[2][2]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['train wins test']+list(np.array([train_test[0][0],train_test[1][0],train_test[2][0]])/len(self.all_packs))+[sum([train_test[0][0],train_test[1][0],train_test[2][0]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['test wins train']+list(np.array([train_test[0][2],train_test[1][2],train_test[2][2]])/len(self.all_packs))+[sum([train_test[0][2],train_test[1][2],train_test[2][2]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['last ties train']+list(np.array([last_train[0][1],last_train[1][1],last_train[2][1]])/len(self.all_packs))+[sum([last_train[0][1],last_train[1][1],last_train[2][1]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['last ties test']+list(np.array([last_test[0][1],last_test[1][1],last_test[2][1]])/len(self.all_packs))+[sum([last_test[0][1],last_test[1][1],last_test[2][1]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['train ties test']+list(np.array([train_test[0][1],train_test[1][1],train_test[2][1]])/len(self.all_packs))+[sum([train_test[0][1],train_test[1][1],train_test[2][1]])/(3*len(self.all_packs))]

            df.to_csv(self.data_path+'/paper/df_perc_times_best.csv', index=False)

        if comparison_type=='in_which_deg_best':
            top_header = [
                "Case",
                "Initialization","Initialization","Initialization",
                "Learning","Learning","Learning",
                "Convergence","Convergence","Convergence",
                "All regions","All regions","All regions"
            ]

            sub_header = [
                "",
                "Low", "Middle", "High",
                "Low", "Middle", "High",
                "Low", "Middle", "High",
                "Low", "Middle", "High",
            ]

            columns = pd.MultiIndex.from_arrays([top_header, sub_header])

            df = pd.DataFrame(columns=columns)

            def plain_list(list):
                return np.array([x for sublista in list for x in sublista])
            
            df.loc[len(df)] =['train wins last']+list(plain_list(last_train[1])/(plain_list(last_train[0])+plain_list(last_train[1])))
            df.loc[len(df)] =['test wins last']+list(plain_list(last_test[1])/(plain_list(last_test[0])+plain_list(last_test[1])))
            df.loc[len(df)] =['test wins train']+list(plain_list(train_test[1])/(plain_list(train_test[0])+plain_list(train_test[1])))

            df.to_csv(self.data_path + '/paper/df_conditioned_deg_perc.csv', index=False)

            df.to_latex(self.data_path + '/paper/df_conditioned_deg_perc.tex',
                index=False,
                escape=False,
                float_format="%.2f",
                multicolumn=True,
                multicolumn_format='c',
                column_format="|c|ccc|ccc|ccc|ccc|"
            )
   
        plt.savefig(self.figure_path+'/analysis/criteria_comparison_'+comparison_type+'.pdf')
        

class AppendixGrapher:

    def __init__(self,data_path,figure_path):

        os.makedirs(data_path+'/paper/', exist_ok=True)
        os.makedirs(figure_path+'/analysis/', exist_ok=True)
        os.makedirs(figure_path+'/appendix/', exist_ok=True)
    
        self.library='SB3'
        self.all_packs=[ 
            # Box2D
            'pack_PPO_LunarLanderContinuous',
            'pack_PPO_BipedalWalker',
            # MuJoCo
            'pack_PPO_Swimmer',
            'pack_PPO_Hopper',
            'pack_PPO_Ant',
            'pack_PPO_HalfCheetah',
            'pack_PPO_Walker2d'        
                            ]
        self.seeds=list(range(1,31))
        self.data_path=data_path
        self.figure_path=figure_path

    def estimations(self,train_white=False):


        def plot_analysis_per_seed(ax,truth,cols_test,cols_train,first_row,last_row,first_column,center_row=False,center_column=False):

            # Estimates with validation
            greens = plt.cm.Greens(np.linspace(0.3, 0.9, len(cols_test.columns)))
            oranges = plt.cm.Oranges(np.linspace(0.3, 0.9, len(cols_train.columns)))

            # Ground truth
            #ax.plot(range(len(truth)),np.array(truth)-np.array(truth), color='black', label='truth')

            for col, color in zip(cols_test.columns[::-1], greens):
                ax.plot(range(len(truth)), abs(np.array(cols_test[col].values)-np.array(truth)), color=color, label=col.split('_')[0]+r' \textit{validation}',linewidth=1)
            for col, color in zip(cols_train.columns[::-1], oranges):
                if train_white:
                    ax.plot(range(len(truth)), abs(np.array(cols_train[col].values)-np.array(truth)), color='white',alpha=0,linewidth=1)
                else:
                    ax.plot(range(len(truth)), abs(np.array(cols_train[col].values)-np.array(truth)), color=color, label=col.split('_')[0]+r' \textit{train}',linewidth=1)


            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            offset = ax.yaxis.get_offset_text()
            offset.set_x(0)

            if first_row:
                ax.set_title(abbreviate(pack.split('_')[2]))
            else:
                ax.yaxis.get_offset_text().set_visible(False)
            if last_row:
                ax.set_xlabel(r'$t$',fontsize=14)
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)
            if first_column and center_row:
                ax.set_ylabel(r'$|\widetilde{f}(\pi_t)-f(\pi_t)|$',fontsize=14)

            if last_row and first_column:
                leg=ax.legend(
                        loc='upper center',
                        bbox_to_anchor=(4, -0.5),
                        ncol=len(ax.get_legend_handles_labels()[0]),
                        frameon=False,fontsize=14,handletextpad=0.5,columnspacing=0.9,handlelength=0.9
                    )
                for line in leg.get_lines():
                    line.set_linewidth(3)

        n_seeds=5

        fig, axs = plt.subplots(n_seeds,len(self.all_packs), figsize=(2.5*len(self.all_packs),1.2*n_seeds),sharex='col',sharey='col')
        plt.subplots_adjust(top=0.95,bottom=0.13,left=0.03,right=0.99, hspace=0.1,wspace=0.2)


        for i,pack in enumerate(self.all_packs):
            df_truth=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')
            df_test=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_all_seq_est.csv')
            df_train=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_train_all_seq_est.csv')

            
            for seed in range(1,n_seeds+1):

                center_row=[True if (seed-1)==n_seeds//2 else False][0]
                first_row=[True if seed==1 else False][0]
                last_row=[True if seed==n_seeds else False][0]
                first_column=[True if i==0 else False][0]

                truth= df_truth[pack+str(seed)].tolist()
                cols_test = df_test[[c for c in df_test.columns if c.endswith('_seed'+str(seed))]]
                cols_test=cols_test.applymap(lambda x: DataConverter.compress_decompress_list(x,compress=False)[0])
                cols_train = df_train[[c for c in df_train.columns if c.endswith('_seed'+str(seed))]]

                plot_analysis_per_seed(axs[seed-1,i],truth,cols_test,cols_train,first_row,last_row,first_column,center_row=center_row)

        if train_white:
            plt.savefig(self.figure_path+'/appendix/estimations_only_test.pdf')
        if not train_white:
            plt.savefig(self.figure_path+'/appendix/estimations.pdf')

    def estimations_selected_conf(self):


        def plot_analysis_per_seed(ax,truth,test,train,first_row,last_row,first_column,center_row):


            # Ground truth
            ax.plot(range(len(train)),np.array(test), color='green', label=r'\textit{validation}',linewidth=2,alpha=0.8)
            ax.plot(range(len(truth)),np.array(truth), color='black', label='truth',linewidth=1)
            ax.plot(range(len(train)),np.array(train), color='orange', label=r'\textit{train}',linewidth=1)
            

            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            offset = ax.yaxis.get_offset_text()
            offset.set_x(0)

            if first_row:
                ax.set_title(abbreviate(pack.split('_')[2]))
            else:
                ax.yaxis.get_offset_text().set_visible(False)
            if last_row:
                ax.set_xlabel(r'$t$',fontsize=14)
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)
            if first_column and center_row:
                ax.set_ylabel(r'$\widetilde{f}(\pi_t)$',fontsize=14)

            if last_row and first_column:
                legend_handles = [
                    mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='ground truth'),
                    mlines.Line2D([], [], color='green', linestyle='-', linewidth=2, label=r'\textit{validation} $\kappa=5$'),
                    mlines.Line2D([], [], color='orange', linestyle='-', linewidth=2, label=r'\textit{train} $\kappa=50$'),
                ]

                leg=ax.legend(
                    handles=legend_handles,
                    loc='upper center',
                    bbox_to_anchor=(4, -0.45),  # fuera de la gráfica (abajo)
                    ncol=3,
                    frameon=False, fontsize=14
                )
                for line in leg.get_lines():
                    line.set_linewidth(3)

        n_seeds=5

        fig, axs = plt.subplots(n_seeds,len(self.all_packs), figsize=(2.5*len(self.all_packs),1.2*n_seeds),sharex='col',sharey='col')
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.1,wspace=0.2)

        # Leer configuraciones generales optimas para poder las marcar en las graficas
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')


        for i,pack in enumerate(self.all_packs):
            df_truth=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')
            df_test=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_all_seq_est.csv')
            df_train=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_train_all_seq_est.csv')

            general_train_n_ep=int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0])
            general_test_n_ep=int(df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0].split('_')[0])
            print(general_test_n_ep,general_train_n_ep)


            for seed in range(1,n_seeds+1):

                first_row=[True if seed==1 else False][0]
                last_row=[True if seed==n_seeds else False][0]
                first_column=[True if i==0 else False][0]
                center_row=[True if seed-1==n_seeds//2 else False][0]

                truth= df_truth[pack+str(seed)].tolist()
                
                test = df_test[str(general_test_n_ep)+'_val_ep_seed'+str(seed)].tolist()
                test=[DataConverter.compress_decompress_list(x,compress=False)[0] for x in test]
                train = df_train[str(general_train_n_ep)+'_traj_ep_seed'+str(seed)].tolist()

                plot_analysis_per_seed(axs[seed-1,i],truth,test,train,first_row,last_row,first_column,center_row=center_row)



        plt.savefig(self.figure_path+'/appendix/estimations_curves.pdf')
    
    def criteria_configurations(self):


        def graph_deg_criteria_conf_by_regions(self,ax1_train,ax2_train,ax3_train,ax1_test,ax2_test,ax3_test,
                                               pack,first_column,prec_metric='relative_perc_criteria_best'):

            # Leer configuraciones generales optimas para poder las marcar en las graficas
            df_conf=pd.read_csv(self.data_path+'/configurations.csv')
            general_train_n_ep=df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]
            conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
            general_test_n_ep=int(conf_str.split('_')[0])
            general_test_freq=float(conf_str.split('_')[1])
            default_train_n_ep=int(df_conf.loc[df_conf["pack"] == pack, "train_default"].iloc[0])

            #----- Cuadricula de grafica interna

            prec1_simple,prec2_simple,prec3_simple=[],[],[]

            # 1) Train conf_prec
            def train_conf_precCI(ax,listas,n_ep_list,nombres=None,optimal_conf=None,title=''):

                for y, data in zip(range(len(listas)), listas):
                    if optimal_conf!=None:
                        if int(optimal_conf)==int(n_ep_list[::-1][len(listas)-1-y]) :
                            ax.axhline(y, color='yellow', linewidth=10,alpha=0.3)
                    if default_train_n_ep==int(n_ep_list[y]):
                        ax.axhline(y, color='black', linewidth=10,alpha=0.05)
                    if general_train_n_ep==int(n_ep_list[::-1][len(listas)-1-y]) :
                        ax.axhline(y, color="red", linewidth=10,alpha=0.3)

                    ax.hlines(y, np.percentile(data, 25), np.percentile(data, 75), color='black')
                    ax.vlines(np.percentile(data, 25), y - 0.2, y + 0.2, color='black')
                    ax.vlines(np.percentile(data, 75), y - 0.2, y + 0.2, color='black')
                    ax.plot(np.median(data), y, 'o', color='black')

                ax.set_xlim(-0.1,1.1)
                ax.grid(axis='x', linestyle='--',alpha=0.4)
                if nombres!=None:
                    ax.set_yticks(range(len(n_ep_list)), n_ep_list)
                    ax.set_yticklabels([rf'${{{n_ep}}}$' for n_ep in n_ep_list])
                    if first_column:
                        ax.set_ylabel(r'\textit{train} $\kappa$',fontsize=14)

                else:
                    ax.set_yticks([])

                ax.set_title(title,fontsize=14)
                ax.set_xticklabels([])
        
            def obtain_best_train_conf(prec1,prec2,prec3,n_ep_list):

                q5_learning=[np.percentile(i, 25)+np.percentile(j, 25)+np.percentile(k, 25) for i,j,k in zip(prec1,prec2,prec3)]
                return n_ep_list[q5_learning.index(max(q5_learning))]
            
            prec1,prec2,prec3,conf_list=from_df_data_to_graph_data(
                [self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv',self.data_path+'/'+pack.replace('pack','SB3')+'/df_train_prec.csv'],pack,which_graph='train_conf_prec',
                prec_metric=prec_metric
            )
    
            n_ep_train=obtain_best_train_conf(prec1,prec2,prec3,conf_list)
            title=['\nInitialization',abbreviate(pack.split('_')[2])+'\nLearning','Convergence']
            train_conf_precCI(ax1_train,prec1,conf_list,nombres=True,optimal_conf=n_ep_train,title=title[0])
            train_conf_precCI(ax2_train,prec2,conf_list,optimal_conf=n_ep_train,title=title[1])
            train_conf_precCI(ax3_train,prec3,conf_list,optimal_conf=n_ep_train,title=title[2])

            conf_list=[int(n_ep) for n_ep in conf_list]
            prec1_simple.append(prec1[conf_list.index(int(general_train_n_ep))])
            prec2_simple.append(prec2[conf_list.index(int(general_train_n_ep))])
            prec3_simple.append(prec3[conf_list.index(int(general_train_n_ep))])

            # 2) Test conf_prec_cost
            def test_conf_precCI_costColor(ax,prec_matrix,cost_matrix,n_ep_list,freq_list,nombres=None,optimal_conf=[None,None]):

                # Grafica
                current_height = 0
                segment_labels = []
                region_centers = []

                for i in range(len(n_ep_list)):
                    
                    region_start = current_height
                    for j in range(len(freq_list)):

                        if int(optimal_conf[0])==int(n_ep_list[i]) and float(optimal_conf[1])==float(freq_list[j]):
                            ax.axhline(current_height, color='yellow', linewidth=10,alpha=0.3)

                        if general_test_n_ep==float(n_ep_list[i]) and general_test_freq==float(freq_list[j]):
                            ax.axhline(current_height, color="red", linewidth=10,alpha=0.3)

                        datos = prec_matrix[i][j]
                           
                        ax.hlines(current_height, np.percentile(datos, 25), np.percentile(datos, 75), color='black')
                        ax.vlines([np.percentile(datos, 25), np.percentile(datos, 75)], current_height-0.2, current_height+0.2, color='black')
                        ax.scatter(np.median(datos), current_height, color='black', marker='o', zorder=3)
                    
                        segment_labels.append(freq_list[j])
                        current_height += 1
                    
                    region_end = current_height - 1
                    region_centers.append((region_start + region_end)/2)
                    
                    if i < len(n_ep_list)-1: # Linea separadora de regiones
                        ax.axhline(current_height-0.5, color='black', linewidth=1)
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                ax.set_yticks(range(len(segment_labels)), segment_labels)
                ax.set_yticklabels([rf'${{{i}}}$' for i in segment_labels])

                if nombres!=None:
                    for center, name in zip(region_centers, n_ep_list):
                        ax.text(-0.25, center, rf'${{{name}}}$',transform=ax.get_yaxis_transform(),ha='right', va='center')
                    if first_column:
                        ax.set_ylabel(r'\textit{validation} $(\kappa,\varphi_c)$',labelpad=35,fontsize=14)
                else:
                    ax.set_yticks([])
      

                ax.set_xlim(-0.1,1.1)
                ax.set_title("")
                ax.invert_yaxis()

            def obtain_best_test_conf(prec1,prec2,prec3,cost1,cost2,cost3,n_ep_list,freq_list,threshold=0.2):

                q5_learning=[]
                for idx_cost in range(len(prec1)):
                    q5_sublist=[]
                    for i,j,k in zip(prec1[idx_cost],prec2[idx_cost],prec3[idx_cost]):
                        q5_sublist.append(np.percentile(i, 25)+np.percentile(j, 25)+np.percentile(k, 25))
                    q5_learning.append(q5_sublist)

                max_cost_learning=[]
                for idx_cost in range(len(cost1)):
                    max_sublist=[]
                    for i,j,k in zip(cost1[idx_cost],cost2[idx_cost],cost3[idx_cost]):
                        max_sublist.append(max(i+j+k))
                    max_cost_learning.append(max_sublist)

               
                i, j = max(((i, j) for i, sub in enumerate(q5_learning) for j, v in enumerate(sub)),key=lambda x: q5_learning[x[0]][x[1]])
                return int(n_ep_list[i]), freq_list[j]

            prec1,prec2,prec3,n_ep_list,freq_list=from_df_data_to_graph_data(
                [self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv',self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_prec.csv'],pack,which_graph='test_conf_prec_cost',
                prec_metric=prec_metric,n_ep_type='with_cost_freq'
            )
            cost1,cost2,cost3,n_ep_list,freq_list=from_df_data_to_graph_data(
                [self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv',self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv'],pack,which_graph='test_conf_prec_cost',
                prec_metric=prec_metric,n_ep_type='with_cost_freq'
            )

            n_ep_test,freq_test=obtain_best_test_conf(prec1,prec2,prec3,cost1,cost2,cost3,n_ep_list,freq_list)
            test_conf_precCI_costColor(ax1_test,prec1,cost1,n_ep_list,freq_list,nombres=True,optimal_conf=[n_ep_test,freq_test])
            test_conf_precCI_costColor(ax2_test,prec2,cost2,n_ep_list,freq_list,optimal_conf=[n_ep_test,freq_test])
            test_conf_precCI_costColor(ax3_test,prec3,cost3,n_ep_list,freq_list,optimal_conf=[n_ep_test,freq_test])


        fig, axs = plt.subplots(8,11, figsize=(1.8*11,2*8),
                                gridspec_kw={'width_ratios': [1]*3+[0.7]+[1]*3+[0.7]+[1]*3,
                                             'height_ratios': [0.1,0.5]+[0.07]+[0.1,0.5]+[0.07]+[0.1,0.5]})
        plt.subplots_adjust(top=0.96,bottom=0.02,left=0.06,right=0.98, hspace=0.1,wspace=0)


        for i,pack in enumerate(self.all_packs):
            first_column=[True if i%3==0 else False][0]
            if i!=len(self.all_packs)-1:
                graph_deg_criteria_conf_by_regions(self,
                                                axs[3*(i//3),4*(i%3)],axs[3*(i//3),1+4*(i%3)],axs[3*(i//3),2+4*(i%3)],
                                                axs[1+3*(i//3),4*(i%3)],axs[1+3*(i//3),1+4*(i%3)],axs[1+3*(i//3),2+4*(i%3)],pack,first_column)
            else:
                graph_deg_criteria_conf_by_regions(self,
                                                axs[6,4],axs[6,5],axs[6,6],
                                                axs[7,4],axs[7,5],axs[7,6],pack,first_column)
            

        for i in range(8):
            axs[i,3].axis('off')
            axs[i,7].axis('off')
        for i in range(11):
            axs[2,i].axis('off')
            axs[5,i].axis('off')

        for i in [0,1,2,8,9,10]:
            axs[6,i].axis('off')
            axs[7,i].axis('off')

        legend_elements = [Line2D([0], [0], color='yellow', lw=6, alpha=0.3, label='max p25_initialization+p25_learning+p25_stabilization'),
                                Line2D([0], [0], color="red", lw=6, alpha=0.3, label='in general (all environments)'), 
                                Line2D([0], [0], color='grey', lw=6, alpha=0.3, label='by default')
                                ]
        axs[6,8].legend(
                    handles=legend_elements,
                    title='',
                    loc='upper center',
                    bbox_to_anchor=(1.5, 0),
                    ncol=1,         
                    frameon=False,
                    fontsize=14
                )


        plt.savefig(self.figure_path+'/appendix/configurations.pdf')

    def cost_of_cost_driven_test(self):

        list_n_ep=[500,250,100,50,25,5]
        list_cost=[0.05,0.1,0.15,0.2]
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        # GRAFICA 1: evolucion de ratio de coste

        def plot_cost_curve_with_CI(ax,df,color,cost,first_column,last_row):
            ax.plot(np.arange(len(df)), df.quantile(0.5, axis=1), color=color,linewidth=1)
            ax.fill_between(np.arange(len(df)),df.quantile(0.25, axis=1),df.quantile(0.75, axis=1),color=color,alpha=0.3)
            ax.set_ylim(0,cost)
            ax.set_yticks([0,cost])
            ax.set_xlim(0,df.shape[0])

            if not first_column:
                ax.tick_params(labelleft=False)
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$')
            if not last_row:
                ax.tick_params(labelbottom=False)
            if last_row:
                ax.set_xlabel(r'$t$')
  
        def plot_iters_upper_cost_threshold(ax,list):

            img = np.array(list, dtype=int)[np.newaxis, :]
            cmap = ListedColormap(["white", "red"])
            ax.imshow(img,aspect='auto',cmap=cmap,interpolation='nearest')

            # limpieza visual
            ax.set_yticks([])
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig, axs = plt.subplots(len(list_cost)*2+len(list_cost)-1,len(self.all_packs), figsize=(1.6*len(self.all_packs),1*len(list_cost)),
                                gridspec_kw={'height_ratios': [0.2, 1,0.1]*(len(list_cost)-1)+[0.2,1]})
        plt.subplots_adjust(top=0.91,bottom=0.15,left=0.07,right=0.98, hspace=0.01,wspace=0.1)

        _, ncols = axs.shape
        for row in [1, 4, 7, 10]:
            ref_ax = axs[row, 0]  # eje referencia de la fila

            for col in range(1, ncols):
                axs[row, col].sharey(ref_ax)

        for i,pack in enumerate(self.all_packs):
            first_column=[True if i==0 else False][0]
            
            df_costs=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')

            for j,cost in enumerate(list_cost):
                last_row=[True if j==len(list_cost)-1 else False][0]
                up_to_cost=[0]*df_costs.shape[0]

                # Primero dibujar curvas de evolucion de coste
                for k,n_ep in enumerate(list_n_ep):

                    df_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{cost}cost_" in c]]

                    # Dibujar curva
                    plot_cost_curve_with_CI(axs[2*j+1+j,i],df_current,colors[k],cost,first_column,last_row)
                    # Mirar en que iteraciones de la secuencia nos hemos pasado del umbral de coste
                    mask = (df_current > cost).any(axis=1)
                    up_to_cost = np.array(up_to_cost)+ np.array(mask.astype(int).tolist())

                # Despues dibujar sobre esta la grafica auxiliar que indica dodnde nos hemos pasado
                plot_iters_upper_cost_threshold(axs[2*j+j,i],up_to_cost)

            axs[0,i].set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=14)

        for i in [2,5,8]:
            for j in range(7):
                axs[i,j].axis('off')

        # Leyenda
        legend_elements = []

        # lineas de colores
        for c, n in zip(colors[:len(list_n_ep)], [ r'$\kappa$='+str(i) for i in list_n_ep]):
            legend_elements.append(Line2D([0], [0], color=c, lw=2, label=str(n)))

        # linea roja extra
        legend_elements.append(Line2D([0], [0], color='red', lw=2, label=r'$\varphi_c$ is surpassed'))
        axs[len(list_cost)*2+len(list_cost)-2,0].legend(handles=legend_elements,
            loc='upper center',bbox_to_anchor=(4, -0.55),ncol=len(legend_elements), frameon=False)
                
        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test.pdf')


        #GRAFICA 2: boxplot de ratio de costes por umbral-setup

        def plot_cost_boxplots_per_pack(ax,data,list_costs,pack,first_column):

            for y in list_cost:
                ax.axhline(y=y, color='gray', linestyle='-', linewidth=1)
            ax.boxplot(
                    data,
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(
                        marker='_',        
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=0.5
                    )
                )
            ax.set_xticks(list(range(1,len(list_costs)+1)))
            ax.set_xticklabels([rf'${{{cost}}}$' for cost in list_costs])
            ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=14)

            if not first_column:
                ax.set_yticks([])
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$',fontsize=12)
            ax.set_xlabel(r'$\varphi_c$',fontsize=12)

        fig, axs = plt.subplots(1,len(self.all_packs), figsize=(1.8*len(self.all_packs),2))
        plt.subplots_adjust(top=0.85,bottom=0.21,left=0.07,right=0.98, hspace=0.1,wspace=0.1)

        for i,pack in enumerate(self.all_packs):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')

            all_ratios_per_cost=[]
            for j,cost in enumerate(list_cost):

                # Guardar todos los ratios de coste para despues dibujar boxplot
                all_ratios=[]
                for k,n_ep in enumerate(list_n_ep):
                    df_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{cost}cost_" in c]]
                    all_ratios+=df_current.values.ravel().tolist()

                # Ratios por coste
                all_ratios_per_cost.append([1 if ratio > 1 else ratio for ratio in all_ratios])

            # Dibujar grafica de boxplots por del pack actual
            plot_cost_boxplots_per_pack(axs[i],all_ratios_per_cost,list_cost,pack,first_column)

        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test_boxplots.png')
        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test_boxplots.pdf')

        # GRAFICA 3: boxplot por pack-umbral-n_ep junto a IQR-median de prec

        def plot_cost_boxplots_per_pack_threshold(ax,data,prec_stats,list_n_ep,pack,cost,first_column,first_row,last_row):

            ax.axhspan(0, cost,color='green',alpha=0.2)
            ax.boxplot(
                    data,
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(
                        marker='_',        
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=0.5
                    )
                )
            for x, (p25, med, p75) in enumerate(prec_stats, start=1):
                ax.fill_between([x - 0.25, x + 0.25],[p25, p25],[p75, p75],color='blue',alpha=0.4,zorder=1)
                ax.plot([x - 0.25, x + 0.25],[med, med],color='blue',linewidth=2,zorder=2)

            ax.set_xticks(list(range(1,len(list_n_ep)+1)))
            ax.set_xticklabels([rf'${{{cost}}}$' for cost in list_n_ep])

            if first_row:
                ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=14)
            if not first_column:
                ax.set_yticks([])
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$',fontsize=12)
                ax.text(-0.5, 0.5,rf'$\varphi_c={cost}$',transform=ax.transAxes,ha='center',va='center',rotation=90,fontsize=12)
            if not last_row:
                ax.set_xticks([])
            if last_row:
                ax.set_xlabel(r'$\kappa$',fontsize=12)

        fig, axs = plt.subplots(len(list_cost),len(self.all_packs), figsize=(1.8*len(self.all_packs),1.5*len(list_cost)))
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.07,right=0.98, hspace=0.1,wspace=0.1)

        for i,pack in enumerate(self.all_packs):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')
            df_precs=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_prec.csv')

            
            for j,cost in enumerate(list_cost):
                first_row=[True if j==0 else False][0]
                last_row=[True if j==len(list_cost)-1 else False][0]

                # Guardar todos los ratios de coste para despues dibujar boxplot
                all_ratios=[]
                all_precs=[]
                for k,n_ep in enumerate(list_n_ep):
                    df_cost_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{cost}cost_" in c]]
                    all_ratios.append([1 if ratio > 1 else ratio for ratio in df_cost_current.values.ravel().tolist()])

                    df_prec_current=df_precs[[c for c in df_precs.columns if f"_{n_ep}_{cost}cost_" in c]]
                    list_prec=df_prec_current.values.ravel().tolist()
                    all_precs.append([np.percentile(list_prec,25),np.median(list_prec),np.percentile(list_prec,75)])


                # Dibujar grafica de boxplots por del pack-threshold actual
                plot_cost_boxplots_per_pack_threshold(axs[j,i],all_ratios,all_precs,list_n_ep,pack,cost,first_column,first_row,last_row)

        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test_boxplots_pack_cost.png')
        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test_boxplots_pack_cost.pdf')


        # GRFACICA 4: boxplot por pack-periodo-n_ep junto a IQR-median de prec para default test criterion
        def plot_cost_boxplots_per_pack_freq(ax,data,prec_stats,list_n_ep,pack,freq,first_column,first_row,last_row):

            ax.axhspan(0, 0.2,color='green',alpha=0.2)
            ax.boxplot(
                    data,
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(
                        marker='_',        
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=0.5
                    )
                )
            for x, (p25, med, p75) in enumerate(prec_stats, start=1):
                ax.fill_between([x - 0.25, x + 0.25],[p25, p25],[p75, p75],color='blue',alpha=0.4,zorder=1)
                ax.plot([x - 0.25, x + 0.25],[med, med],color='blue',linewidth=2,zorder=2)
            ax.set_xticks(list(range(1,len(list_n_ep)+1)))
            ax.set_xticklabels([rf'${{{cost}}}$' for cost in list_n_ep])

            if first_row:
                ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=14)
            if not first_column:
                ax.set_yticks([])
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$',fontsize=12)
                ax.text(-0.5, 0.5,rf'$\varphi={freq}$',transform=ax.transAxes,ha='center',va='center',rotation=90)
            if not last_row:
                ax.set_xticks([])
            if last_row:
                ax.set_xlabel(r'$\kappa$',fontsize=12)

        list_freq=[20,10,5,1]

        fig, axs = plt.subplots(len(list_freq),len(self.all_packs), figsize=(1.8*len(self.all_packs),1.5*len(list_freq)))
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.07,right=0.98, hspace=0.1,wspace=0.1)

        for i,pack in enumerate(self.all_packs):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')
            df_precs=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_prec.csv')

            
            for j,freq in enumerate(list_freq):
                first_row=[True if j==0 else False][0]
                last_row=[True if j==len(list_freq)-1 else False][0]

                # Guardar todos los ratios de coste para despues dibujar boxplot
                all_ratios=[]
                all_precs=[]
                for k,n_ep in enumerate(list_n_ep):
                    df_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{freq}_" in c]]
                    all_ratios.append([1 if ratio > 1 else ratio for ratio in df_current.values.ravel().tolist()])

                    df_prec_current=df_precs[[c for c in df_precs.columns if f"_{n_ep}_{freq}_" in c]]
                    list_prec=df_prec_current.values.ravel().tolist()
                    all_precs.append([np.percentile(list_prec,25),np.median(list_prec),np.percentile(list_prec,75)])


                # Dibujar grafica de boxplots por del pack-threshold actual
                plot_cost_boxplots_per_pack_freq(axs[j,i],all_ratios,all_precs,list_n_ep,pack,freq,first_column,first_row,last_row)

        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test_boxplots_pack_freq.png')
        plt.savefig(self.figure_path+'/appendix/cost_of_cost_driven_test_boxplots_pack_freq.pdf')

    def val_cost_throughout_sequence(self):

        # Escala de colores
        cmap_gwr = LinearSegmentedColormap.from_list('green_white_red',
            [(0.0, 'darkgreen'),(0.3, 'white'),(1.0, 'darkred')])

        norm_gwr = Normalize(vmin=0, vmax=1)

        def plot_analysis_per_seed(ax1,ax2,axs_cost,list1,list2,df,last_row,pack):
            
            def normalize(x):
                x = np.array(x, dtype=float)
                if x.max() == x.min():
                    return np.array([1]*len(x))
                return (x - x.min()) / (x.max() - x.min())

            # Normalizar truth y ep_lens
            list1_norm = normalize(list1)
            list2_norm = normalize(list2)

            # Grafica
            #--- truth
            ax1.imshow(list1_norm[np.newaxis, :],cmap='gray_r',aspect='auto',interpolation='nearest',vmin=0,vmax=1)
            ax1.set_ylabel(r'$f(\pi_t)$', rotation=0,fontsize=8)
            ax1.yaxis.set_label_coords(-0.05, 0)

            #--- ep_lens
            ax2.imshow(list2_norm[np.newaxis, :],cmap='gray_r',aspect='auto',interpolation='nearest',vmin=0,vmax=1)
            ax2.set_ylabel(r'$|\tau|$', rotation=0,fontsize=8)
            ax2.yaxis.set_label_coords(-0.05, 0)

            #--- val_cost
            for i, col in enumerate(df.columns):
                values = np.array(df[col].values, dtype=float)

                # limitar valores >1 a 1 para el color
                values_clip = np.clip(values, 0, 1)
                
                axs_cost[i].imshow(values_clip[np.newaxis, :],cmap=cmap_gwr,norm=norm_gwr,aspect='auto',interpolation='nearest')
                axs_cost[i].set_ylabel(rf'${{{col.split('_')[0]}}}$', rotation=0,fontsize=8)
                axs_cost[i].yaxis.set_label_coords(-0.05, 0)

            # Limpieza visual
            all_axs=[ax1,ax2]+list(axs_cost)
            for ax in all_axs:
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if last_row:
                for ax in all_axs[:-1]:
                    ax.set_xticks([])
                all_axs[-1].set_xlabel(r'$t$',fontsize=10)
            else:
                for ax in all_axs[:-2]:
                    ax.set_xticks([])
                all_axs[-1].set_xlabel('')
            all_axs[-2].tick_params(axis='x', labelsize=8,length=0.5)
            all_axs[-1].tick_params(axis='x', labelsize=8,length=0.5)
            axs_cost[1].text(-0.1, 0.5,abbreviate(pack.replace('pack_PPO_','')),rotation=90,va='center',ha='center',transform=axs_cost[1].transAxes,fontsize=10)


  
        fig, axs = plt.subplots(8*len(self.all_packs)+len(self.all_packs)-1,1, figsize=(5,1*len(self.all_packs)),
                                gridspec_kw={'height_ratios': [1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8})
        plt.subplots_adjust(top=0.99,bottom=0.13,left=0.1,right=0.98, hspace=0.1,wspace=0.2)

        what_seed=1

        for i,pack in enumerate(self.all_packs):
            df_truth=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')
            df_test=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_all_seq_cost_len.csv')
               
            truth=df_truth[pack+str(what_seed)].tolist()
            ep_lens=df_test['ep_len_seed'+str(what_seed)].tolist()
            val_costs=df_test[[c for c in df_test.columns if c.endswith('val_ep_seed'+str(what_seed))]]

            last_row=[True if i==len(self.all_packs)-1 else False][0]
            plot_analysis_per_seed(axs[9*i],axs[1+9*i],axs[2+9*i:9+9*i],truth[1:],ep_lens[1:],val_costs.iloc[1:],last_row,pack)

        for i in [8,17,26,35,44,53]:
            axs[i].axis('off')

        # Leyenda
        sm = plt.cm.ScalarMappable(cmap=cmap_gwr, norm=norm_gwr)
        sm.set_array([])
        cmap_bw = LinearSegmentedColormap.from_list("white_black", ["white", "black"])
        sm2 = plt.cm.ScalarMappable(cmap=cmap_bw, norm=plt.Normalize(0, 1))
        sm2.set_array([])

        # posición base (ajústalo si quieres subir/bajar todo)
        y = 0.06
        h = 0.01
        w = 0.35

        # izquierda
        cax1 = fig.add_axes([0.08+0.05, y, w, h])
        cbar1 = fig.colorbar(sm, cax=cax1, orientation='horizontal')
        cbar1.set_label('Validation-to-learning iter. cost ratio', fontsize=10)
        cbar1.ax.tick_params(labelsize=8)

        # derecha
        cax2 = fig.add_axes([0.54+0.05, y, w, h])
        cbar2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
        cbar2.set_label(r'Normalized $f(\pi_t)$ or mean $|\tau|$', fontsize=10)
        cbar2.ax.tick_params(labelsize=8)

        plt.savefig(self.figure_path+'/appendix/val_cost_throughout_sequence.pdf')

    def val_freq_throughout_sequence(self):

        def plot_val_freq_evolution(ax,df,pack,last_row):

            img = (df.to_numpy().T > 0).astype(int)
            cmap = ListedColormap(["white", "black"])
            ax.imshow(img,cmap=cmap,aspect='auto',vmin=0,vmax=1,interpolation='nearest')
            ax.set_yticks([])
            ax.set_ylabel(abbreviate(pack.replace('pack_PPO_','')),fontsize=10)
            if last_row:
                ax.set_xlabel(r'$t$',fontsize=10)

            ax.tick_params(axis='x', labelsize=8,length=0.5)


        fig, axs = plt.subplots(len(self.all_packs),1, figsize=(5,1*len(self.all_packs)))
        plt.subplots_adjust(top=0.99,bottom=0.05,left=0.05,right=0.98, hspace=0.25,wspace=0.2)


        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        test_conf=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        n_ep=test_conf.split('_')[0]
        cost=test_conf.split('_')[1]

        for i,pack in enumerate(self.all_packs):
            last_row=[True if i==len(self.all_packs)-1 else False][0]

            # Columnas de la base de datos de interes
            df_n_ep=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_n_ep.csv')
            df_current=df_n_ep[[c for c in df_n_ep.columns if f"_{n_ep}_{cost}cost_" in c]]

            # Dibujar grafica por pack de las 30 semillas
            
            plot_val_freq_evolution(axs[i],df_current,pack,last_row)


        plt.savefig(self.figure_path+'/appendix/val_freq_throughout_sequence.pdf')

    def test_default_vs_with_cost(self):

        # Configuracion de test with cost
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        # Funciones
        def plot_pack_costs_accs(ax,cost_list,pack_name='',ytitle='',title='',not_last_pack=True,first_pack=False,first_column=True):
 
            # Nube de puntos
            sns.stripplot(x=cost_list, orient='h', jitter=0.4,ax=ax,color='black',zorder=1,size=1)

            # Linea de la mediana
            ax.axvline(np.median(cost_list), color='red',linewidth=2,zorder=2)
            ax.axvspan(np.quantile(cost_list,0.25), np.quantile(cost_list,0.75), color='red', alpha=0.5,zorder=2)

            ax.set_xlim(-0.05, 1.05)
            ax.text(-0.4, -2.5, ytitle,transform=ax.transAxes,rotation=90,va="center",ha="center",fontsize=12)
            if first_column:
                ax.set_ylabel(abbreviate(pack_name), rotation=0,fontsize=10)
                ax.yaxis.set_label_coords(-0.15, 0.1) 

            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'],fontsize=12)

            if not_last_pack:   
                ax.tick_params(axis='x', labelbottom=False)
            if first_pack:
                ax.set_title(title,fontsize=12)
            ax.set_yticks([]) 

        def plot_val_times(ax,times_list,single_or_list='list',pack_name='',not_last_pack=True,first_column=True,first_pack=False):

            if single_or_list=='list':
                ax.barh(0, np.median(times_list), height=0.3, color='black')
                ax.axvspan(np.quantile(times_list,0.25), np.quantile(times_list,0.75), color='red', alpha=0.5,zorder=2)

            if single_or_list=='number':
                ax.barh(0, times_list, height=0.3, color='black')
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'],fontsize=12)
            
            if first_column:
                ax.set_ylabel(abbreviate(pack_name), rotation=0,fontsize=10)
                ax.yaxis.set_label_coords(-0.15, 0.1) 
            if not_last_pack:   
                ax.tick_params(axis='x', labelbottom=False)
            if first_pack:
                ax.set_title('Proportion of\nvalidations',fontsize=12)
            ax.set_yticks([]) 
                
        # Grafica
        fig, axs = plt.subplots(len(self.all_packs)*2+1,5, figsize=(3*2,0.4*len(self.all_packs)),
                                    gridspec_kw={'width_ratios': [1,0,1,0,1],
                                                 'height_ratios': [1]*len(self.all_packs)+[0.5]+[1]*len(self.all_packs)})
        plt.subplots_adjust(top=0.86,bottom=0.1,left=0.15,right=0.98, hspace=0.0,wspace=0.0)

        axs = np.atleast_2d(axs)

        for i,pack in enumerate(self.all_packs):

            
            # Leear bases de datos necesarias
            df_cost=pd.read_csv(self.data_path+'/'+pack.replace('pack',self.library)+'/df_test_cost.csv')
            df_n_ep=pd.read_csv(self.data_path+'/'+pack.replace('pack',self.library)+'/df_test_n_ep.csv')
            df_acc=pd.read_csv(self.data_path+'/'+pack.replace('pack',self.library)+'/df_test_prec.csv')

            # Listas de costes
            test_default=df_conf.loc[df_conf["pack"] == pack, "test_default"].iloc[0]
            df_default= df_cost.loc[:, df_cost.columns.str.contains('_'+test_default+'_')]
            cost_default = [min(cost, 1) for cost in df_default.values.flatten().tolist()] 
            df_with_cost=df_cost.loc[:, df_cost.columns.str.contains('_'+test_conf+'_')]
            cost_with_cost=[min(cost, 1) for cost in df_with_cost.values.flatten().tolist()] 

            # Listas de numero de validaciones
            times_default=int(df_n_ep.shape[0]/int(test_default.split('_')[1]))/df_n_ep.shape[0]
            df_with_cost=df_n_ep.loc[:, df_n_ep.columns.str.contains('_'+test_conf+'_')]
            times_with_cost=[i/df_n_ep.shape[0] for i in (df_with_cost != 0).sum().tolist()]

            # Listas de accuracies
            df_default= df_acc.loc[:, df_acc.columns.str.contains('_'+test_default+'_')]
            acc_default=[acc for acc in df_default.values.flatten().tolist()] 
            df_default= df_acc.loc[:, df_acc.columns.str.contains('_'+test_conf+'_')]
            acc_with_cost=[acc for acc in df_default.values.flatten().tolist()] 


            not_last_pack=[False if i==len(self.all_packs)-1 else True][0]
            first_pack=[True if i==0 else False][0]
            titles=[[r'\noindent Default\\ \textit{validation}', r'\noindent Cost-driven\\\textit{validation}']if i==0 else ['','']][0]
            plot_pack_costs_accs(axs[i,0],cost_default,pack_name=pack.replace('pack_PPO_',''),ytitle=titles[0],not_last_pack=True,first_pack=first_pack,title='Validation-to-learning\ncost ratio')
            plot_pack_costs_accs(axs[i+len(self.all_packs)+1,0],cost_with_cost,pack_name=pack.replace('pack_PPO_',''),ytitle=titles[1],not_last_pack=not_last_pack)
            plot_val_times(axs[i,2],times_default,pack_name=pack.replace('pack_PPO_',''),single_or_list='number',not_last_pack=True,first_column=False,first_pack=first_pack)
            plot_val_times(axs[i+len(self.all_packs)+1,2],times_with_cost,pack_name=pack.replace('pack_PPO_',''),single_or_list='list',not_last_pack=not_last_pack,first_column=False)
            plot_pack_costs_accs(axs[i,4],acc_default,pack_name=pack.replace('pack_PPO_',''),not_last_pack=True,first_column=False,first_pack=first_pack,title='Accuracies\n')
            plot_pack_costs_accs(axs[i+len(self.all_packs)+1,4],acc_with_cost,pack_name=pack.replace('pack_PPO_',''),not_last_pack=not_last_pack,first_column=False)

        for i in range(len(self.all_packs)):
            axs[i,1].axis('off')
            axs[i,3].axis('off')
            axs[len(self.all_packs)+i+1,1].axis('off')
            axs[len(self.all_packs)+i+1,3].axis('off')

        for i in range(5):
            axs[len(self.all_packs),i].axis('off')


        plt.savefig(self.figure_path+'/appendix/test_default_vs_with_cost.png')
        plt.savefig(self.figure_path+'/appendix/test_default_vs_with_cost.pdf')

    def regions_with_deg(self):

        def plot_analysis_per_seed(ax1,ax2,pack,seed,first_row=False,first_column=False,last_row=False,center_row=False,
                                   global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff'):

            colors=list(mcolors.TABLEAU_COLORS.keys())

            # Evolucion de degradacion
            deg=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/deg_evolution.csv')[pack+str(seed)+'_'+global_deg_metric+'_'+local_deg_metric].tolist()
            
            ax1.imshow(np.array(deg)[np.newaxis, :], cmap='gray_r', vmin=0, vmax=1,aspect='auto', interpolation='nearest')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Evolucion de estimaciones truth
            truth_last=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')[pack+str(seed)].tolist()
            ax2.plot(range(len(truth_last)), truth_last, label="Truth",color=colors[0],linewidth=1)

            # Limites a (donde empieza a aprender) y b (donde se termina de aprender)
            df_limits=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv')
            a,b=df_limits.loc[(df_limits['pack_seed'] == pack + str(seed)),['a', 'b']].iloc[0]

            ax2.axvline(x=a, color='red', linestyle='-', linewidth=2,label='start learning')
            ax2.axvline(x=b, color='black', linestyle='-', linewidth=2,label='end learning')
            ax2.axvline(x=int(len(truth_last)/1.5), color='black', linestyle='--', linewidth=2,label='default end')
            ax2.set_xlim(0,len(truth_last))
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            offset = ax2.yaxis.get_offset_text()
            offset.set_x(-0.19)

            if first_row:
                ax1.set_title(abbreviate(pack.split('_')[2]),fontsize=14)
            if last_row:
                ax2.set_xlabel(r'$t$',fontsize=12)
                ax2.tick_params(labelbottom=True)
            else:
                ax2.tick_params(labelbottom=False)
            if first_column and center_row:
                ax2.set_ylabel(r'$f(\pi_t)$',fontsize=12)

        n_seeds=5

        fig, axs = plt.subplots(n_seeds*2,len(self.all_packs), figsize=(2.5*len(self.all_packs),1.2*n_seeds),
                                height_ratios=[0.2,1]*n_seeds)
        plt.subplots_adjust(top=0.95,bottom=0.13,left=0.04,right=0.99, hspace=0.2,wspace=0.2)

        # compartir ejes solo para filas impares
        for col in range(len(self.all_packs)):
            
            # referencia = primera fila impar
            ref_ax = axs[1, col]
            
            for row in range(3, n_seeds*2, 2):
                axs[row, col].sharex(ref_ax)
                axs[row, col].sharey(ref_ax)

        for i,pack in enumerate(self.all_packs):
            for seed in range(1,n_seeds+1):#TODO: puede que para meter en el paper tenga que reducir el numero de semillas, igual a 5
                first_row=[True if seed==1 else False][0]
                last_row=[True if seed==n_seeds else False][0]
                first_column=[True if i==0 else False][0]
                center_row=[True if seed-1==n_seeds//2 else False][0]
                j=seed-1
                plot_analysis_per_seed(axs[j*2-n_seeds*2*((j+1)//n_seeds),i],axs[j*2+1-n_seeds*2*((j+1)//n_seeds),i],pack,seed,
                                       first_row=first_row,last_row=last_row,first_column=first_column,center_row=center_row)
                
        legend_elements = [
                    Patch(facecolor='black', edgecolor='black', label=r'$\delta_t=1$'),
                    Patch(facecolor='white', edgecolor='black', label=r'$\delta_t=0$'),
                    Line2D([0], [0], color='red', lw=2, label=r'$a$'),
                    Line2D([0], [0], color='black', lw=2, label=r'$b$'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', label=r'$t_{max}$')
                    
                ]
        fig.legend(handles=legend_elements,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.5, -0.01),fontsize=12)

        plt.savefig(self.figure_path+'/appendix/regions.pdf')    

    def criteria_consequences(self,consequence_type):

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]

        legend_elements = [Line2D([0], [0], color='black', label='Ground-truth best', linestyle='-', linewidth=1.5),
                            Line2D([0], [0], marker='o', color='blue', label=r'\textit{Last}',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='^', color='orange', label=r'Default \textit{train}',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='D', color="#A52D81", label=r'Default \textit{validation}',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='s', color='green', label=r'Cost-driven \textit{validation}',markersize=6, linestyle='')       
                        ]

        def plot_pack_early_stopping(axs,pack_name,pack_indx,first_pack=False,early_stopping_graph='all'):

            test_n_ep_default,test_freq_default=int(test_default.split('_')[0]),int(test_default.split('_')[1])
            test_n_ep,test_freq=int(conf_str.split('_')[0]),float(conf_str.split('_')[1])

            # Obtener datos para las graficas
            matrix_best_est,matrix_best_truth=successive_halving(self.data_path+'/',pack)
            matrix_last_est,matrix_last_truth=successive_halving(self.data_path+'/',pack,criteria='last')
            matrix_train_default_est,matrix_train_default_truth=successive_halving(self.data_path+'/',pack,criteria='train',conf=[train_default,None])
            matrix_test_default_est,matrix_test_default_truth=successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep_default,test_freq_default])
            matrix_test_est,matrix_test_truth=successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep,test_freq])

            
            # Grafica 1: sucessive halving explicito
            def plot_succesive_halving(ax,matrix_criteria,color):
                max_long=max([len(i) for i in matrix_criteria])
                for truth_evol in matrix_criteria:
                    ax.plot(range(len(truth_evol)) , truth_evol, color=color,linewidth=1)  
                    if len(truth_evol)!=max_long:
                        ax.axvline(x=len(truth_evol), color='red', linewidth=1)

                    ax.tick_params(axis='both', labelsize=12)

            if early_stopping_graph=='all':
                plot_succesive_halving(axs[0,pack_indx],matrix_best_truth,'black')
                plot_succesive_halving(axs[1,pack_indx],matrix_last_truth,'blue')
                plot_succesive_halving(axs[2,pack_indx],matrix_train_default_truth,'orange')
                plot_succesive_halving(axs[3,pack_indx],matrix_test_default_truth,'purple')
                plot_succesive_halving(axs[4,pack_indx],matrix_test_truth,'green')

                if first_pack:
                    for i,ylabel in enumerate(['Ground truth',r'\textit{Last}',r'Default \textit{train}',r'Default \textit{validation}',r'Cost-driven \textit{validation}']):
                        axs[i,pack_indx].set_ylabel(ylabel+'\n'+r"$f(\widetilde{\pi}_t)$",fontsize=12)
                for i in range(5):
                    axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                    if i!=0:
                        axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)

                
                axs[0,pack_indx].set_title(abbreviate(pack_name),fontsize=14)
                axs[4,pack_indx].set_xlabel(r"$t$",fontsize=12)

            # Graficas 2 y 3: info de resource allocation (eff) y evolucion de performnace (prec)
            def plot_resource_allocation(ax, listas, colors,center_column=False):

                block_centers = []

                # markers por color (orden fijo)
                marker_map = {'blue': 'o','orange': '^','green': 's','purple': 'D'}

                # 1. rango global
                all_data = [x for lista in listas for sub in lista for x in sub]
                min_val = min(all_data)
                max_val = max(all_data)

                # bins no uniformes
                bins = np.array([min_val,min_val + 0.25 * (max_val - min_val),min_val + 0.75 * (max_val - min_val),max_val])

                n_groups = len(listas)
                bar_height = (max_val - min_val) * 0.06
                spacing = bar_height * 1.15
                outer_margin = (max_val - min_val) * 0.08

                # 2. histogramas
                histogramas = []
                for lista_de_listas in listas:
                    datos = [x for sub in lista_de_listas for x in sub]
                    counts, _ = np.histogram(datos, bins=bins)
                    perc = counts / len(datos)
                    histogramas.append(perc)

                # 3. dibujar
                for b in range(len(bins) - 1):

                    y_center = (bins[b] + bins[b+1]) / 2
                    block_centers.append(y_center)

                    total_height = (n_groups - 1) * spacing
                    start = y_center - total_height / 2

                    for i, counts in enumerate(histogramas):
                        y = start + i * spacing
                        ax.barh(y,counts[b],height=bar_height,left=0,color=colors[i],align='center',alpha=0.8)

                        color = colors[i]
                        if color in marker_map and color != 'black':
                            ax.scatter(counts[b],y,marker=marker_map[color],color=colors[i],edgecolor='black',zorder=5,s=30)

                # líneas separadoras entre bloques
                for i in range(len(block_centers) - 1):

                    y_sep = (block_centers[i] + block_centers[i+1]) / 2

                    ax.axhline(
                        y=y_sep,
                        color='black',
                        linewidth=0.8,
                        zorder=10
                    )

                # 4. estetica
                ax.set_ylim(min_val - outer_margin, max_val + outer_margin)  
                ax.set_xlim(0, max(max(h) for h in histogramas) * 1.1)

                bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
                ax.set_yticks(bin_centers)

                if center_column:
                    ax.set_xlabel('Proportion of evaluations',fontsize=12)

                # para que los numeros del eje x no se superpongan
                step = 100

                xmin, xmax = ax.get_xlim()

                tick_1 = step * round((xmin + (xmax - xmin)/3) / step,3)
                tick_2 = 2 * tick_1

                ax.set_xticks([0, tick_1, tick_2])
                ax.set_xticklabels([r'$0$', rf'${{{tick_1}}}$', rf'${{{tick_2}}}$'],fontsize=12)
                        
                if pack_indx == 0:
                    ax.set_ylabel('Truth reward values',fontsize=12)
                    ax.set_yticklabels(["low", "middle", "high"],fontsize=12)
                    ax.legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(3.5, -0.2),ncol=5,frameon=False,fontsize=12)
                else:
                    ax.set_yticklabels(["", "", ""])    

            # def plot_resource_allocation(ax, listas, colors):

            #     mpl.rcParams['hatch.linewidth'] = 0.6
            #     block_centers = []

            #     # textura por color
            #     hatch_map = {
            #         'blue': '..',
            #         'orange': '//',
            #         'green': '||',
            #         'purple': 'xx',
            #         'black': None
            #     }

            #     # 1. rango global
            #     all_data = [x for lista in listas for sub in lista for x in sub]
            #     min_val = min(all_data)
            #     max_val = max(all_data)

            #     bins = np.array([
            #         min_val,
            #         min_val + 0.25 * (max_val - min_val),
            #         min_val + 0.75 * (max_val - min_val),
            #         max_val
            #     ])

            #     n_groups = len(listas)
            #     bar_height = (max_val - min_val) * 0.06
            #     spacing = bar_height * 1.2
            #     outer_margin = (max_val - min_val) * 0.08

            #     # 2. histogramas
            #     histogramas = []
            #     for lista_de_listas in listas:
            #         datos = [x for sub in lista_de_listas for x in sub]
            #         counts, _ = np.histogram(datos, bins=bins)
            #         histogramas.append(counts)

            #     # 3. dibujar
            #     for b in range(len(bins) - 1):

            #         y_center = (bins[b] + bins[b+1]) / 2
            #         block_centers.append(y_center)

            #         total_height = (n_groups - 1) * spacing
            #         start = y_center - total_height / 2

            #         for i, counts in enumerate(histogramas):

            #             y = start + i * spacing

            #             ax.barh(
            #                 y,
            #                 counts[b],
            #                 height=bar_height,
            #                 left=0,
            #                 color=colors[i],
            #                 hatch=hatch_map.get(colors[i], None),
            #                 edgecolor='black' if colors[i] != 'black' else 'black',
            #                 linewidth=0
            #             )

            #     # líneas separadoras entre bloques
            #     for i in range(len(block_centers) - 1):

            #         y_sep = (block_centers[i] + block_centers[i+1]) / 2

            #         ax.axhline(
            #             y=y_sep,
            #             color='black',
            #             linewidth=0.8,
            #             zorder=10
            #         )

            #     # 4. estética
            #     ax.set_ylim(min_val - outer_margin, max_val + outer_margin)
            #     ax.set_xlim(0, max(max(h) for h in histogramas) * 1.1)

            #     bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            #     ax.set_yticks(bin_centers)

            #     ax.set_xlabel('Number of evaluations')

            #     step = 100
            #     xmin, xmax = ax.get_xlim()

            #     tick_1 = step * round((xmin + (xmax - xmin)/3) / step)
            #     tick_2 = 2 * tick_1

            #     ax.set_xticks([0, tick_1, tick_2])

            #     if pack_indx == 0:
            #         ax.set_ylabel('Truth values')
            #         ax.set_yticklabels(["low", "middle", "high"])


            #         legend_elements = [
            #             mpatches.Patch(
            #                 facecolor='black',
            #                 edgecolor='black',
            #                 label='truth best'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='blue',
            #                 edgecolor='black',
            #                 hatch='..',
            #                 label='last'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='orange',
            #                 edgecolor='black',
            #                 hatch='//',
            #                 label='train'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='purple',   # granate aproximado
            #                 edgecolor='black',
            #                 hatch='xx',
            #                 label='default test'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='green',
            #                 edgecolor='black',
            #                 hatch='||',
            #                 label='cost-driven test'
            #             )
            #         ]

            #         ax.legend(
            #             handles=legend_elements,
            #             loc='upper center',
            #             bbox_to_anchor=(1.8, -0.25),
            #             ncol=5,
            #             frameon=False,
            #             handleheight=1.2
            #         )
            #     else:
            #         ax.set_yticklabels(["", "", ""])

            def accuracy_evol(matrix_criteria_est,matrix_criteria_truth,plot_yes=False,
                                       curve_truth=None,curve_criteria=None,current_min=None):

                    
                if plot_yes:
                    acc_evol=[]
                    for i in range(len(curve_truth)):
                        numerator=curve_criteria[i]-current_min[i]
                        denominator=curve_truth[i]-current_min[i]
                        
                        acc_evol.append([1 if denominator==0 or curve_criteria[i]>curve_truth[i] else numerator/denominator])

                    return acc_evol
                else:

                    max_long=max([len(i) for i in matrix_criteria_truth])

                    current_best=[]
                    for i in range(max_long):
                        best_by_process_est=[]
                        best_by_process_truth=[]
                        for sub_est,sub_truth in zip(matrix_criteria_est,matrix_criteria_truth):
                            if len(sub_truth)>=i+1:
                                best_by_process_est.append(sub_est[i])
                                best_by_process_truth.append(sub_truth[i])
                        current_best.append(best_by_process_truth[best_by_process_est.index(max(best_by_process_est))]) 

                    return current_best
                
            def get_current_min(matrix_truth,matrix_criteria):

                max_long=max([len(i) for i in matrix_truth])
                current_min=[]
                for i in range(max_long):
                    # Minimo global para normalizar despues
                    current_truht_mins,current_criteria_mins=[],[]
                    for sub_truth,sub_criteria in zip(matrix_truth,matrix_criteria):
                        current_truht_mins.append(min(sub_truth[:i+1]))
                        current_criteria_mins.append(min(sub_criteria[:i+1]))
                    current_min.append(min(current_truht_mins+current_criteria_mins))

                return current_min

            def plot_pack_acc_CIs(ax,acc_lists,pack,first_pack=False,xlabel=False):

                colors=['blue','orange',"#A52D81",'green']
                markers=['o','^','D','s']

                for i, data in enumerate(acc_lists):

                    p25 = np.percentile(data, 25)
                    med = np.median(data)
                    p75 = np.percentile(data, 75)

                    ax.hlines(i, p25, p75, color=colors[i], linewidth=1)
                    ax.vlines([p25, p75], i-0.15, i+0.15, color=colors[i], linewidth=1)
                    ax.plot(med, i, markers[i], color=colors[i], markersize=6)

                    ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=14)
                    ax.set_xlim(-0.1,1.1)
                    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                    ax.set_xticklabels([r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'])
                    ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.6)
                    if first_pack:
                        ax.set_yticks(range(len(acc_lists)))
                        ax.set_yticklabels([r'\textit{Last}', r'\noindent Default\\\textit{train}', r'\noindent Default\\\textit{validation}', r'\noindent Cost-driven\\\textit{validation}'],fontsize=12)
                    else:
                        ax.set_yticklabels([]) 
                        ax.tick_params(axis='y', which='both', left=True, right=False, length=5)

                    if xlabel:
                        ax.set_xlabel(r'$\alpha^{SH}$',fontsize=12)

                    ax.tick_params(axis='x', labelsize=10)

            if early_stopping_graph=='eff':
                center_column=[True if pack_indx==len(self.all_packs)//2 else False][0]
                plot_resource_allocation(axs[pack_indx],
                                 [matrix_best_truth,matrix_last_truth,matrix_train_default_truth,matrix_test_default_truth,matrix_test_truth],
                                 ['black','blue','orange','purple','green'],center_column=center_column)
                axs[pack_indx].set_title(abbreviate(pack_name),fontsize=14)

            if early_stopping_graph=='prec':
  
                truth_curve=accuracy_evol(matrix_best_est,matrix_best_truth)
                last_curve=accuracy_evol( matrix_last_est,matrix_last_truth)
                traind_curve=accuracy_evol( matrix_train_default_est,matrix_train_default_truth)
                testd_curve=accuracy_evol( matrix_test_default_est,matrix_test_default_truth)
                test_curve=accuracy_evol( matrix_test_est,matrix_test_truth)

                current_min=get_current_min(matrix_best_truth,matrix_best_truth)
                accuracy_evol( matrix_best_est,matrix_best_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=truth_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_last_truth)
                acc_last=accuracy_evol( matrix_last_est,matrix_last_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=last_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_train_default_truth)
                acc_traind=accuracy_evol( matrix_train_default_est,matrix_train_default_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=traind_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_test_default_truth)
                acc_testd=accuracy_evol( matrix_test_default_est,matrix_test_default_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=testd_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_test_truth)
                acc_test=accuracy_evol( matrix_test_est,matrix_test_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=test_curve,current_min=current_min)

                first_pack=[True if pack_indx==0 else False][0]
                xlabel=[True if pack_indx==len(self.all_packs)//2 else False][0]
                plot_pack_acc_CIs(axs[pack_indx],[acc_last,acc_traind,acc_testd,acc_test],pack,first_pack=first_pack,xlabel=xlabel)

        if consequence_type=='early_stopping_all':
            fig, axs = plt.subplots(5,len(self.all_packs), figsize=(2.5*len(self.all_packs),9),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.95,bottom=0.05,left=0.06,right=0.98, hspace=0.05,wspace=0.25)
        if consequence_type =='early_stopping_eff':
            fig, axs = plt.subplots(1,len(self.all_packs), figsize=(2*len(self.all_packs),3))
            plt.subplots_adjust(top=0.9,bottom=0.25,left=0.07,right=0.98, hspace=0.0,wspace=0.05)
        if consequence_type in['early_stopping_prec']:
            fig, axs = plt.subplots(1,len(self.all_packs), figsize=(2*len(self.all_packs),2.2))
            plt.subplots_adjust(top=0.85,bottom=0.18,left=0.08,right=0.99, hspace=0.0,wspace=0.05)

        for i,pack in enumerate(self.all_packs):

            first_pack=[True if i==0 else False][0]
            train_default=int(df_conf.loc[df_conf["pack"] == pack, "train_default"].iloc[0])
            test_default=df_conf.loc[df_conf["pack"] == pack, "test_default"].iloc[0]

            if consequence_type=='early_stopping_all':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack)
            if consequence_type =='early_stopping_eff':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack,early_stopping_graph='eff')
            if consequence_type =='early_stopping_prec':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack,early_stopping_graph='prec')
            
        plt.savefig(self.figure_path+'/appendix/criteria_consequences_'+consequence_type+'.pdf')

    def resource_allocation(self,test_n_ep=5,list_test_freq=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]):

        def accuracy_curve(matrix_criteria_est,matrix_criteria_truth,plot_yes=False,
                                       curve_truth=None,curve_criteria=None,current_min=None):

                if plot_yes:
                    acc_evol=[]
                    for i in range(len(curve_truth)):
                        numerator=curve_criteria[i]-current_min[i]
                        denominator=curve_truth[i]-current_min[i]
                        
                        acc_evol.append([1 if denominator==0 or curve_criteria[i]>curve_truth[i] else numerator/denominator][0])

                    return acc_evol
                else:

                    max_long=max([len(i) for i in matrix_criteria_truth])

                    current_best=[]
                    for i in range(max_long):
                        best_by_process_est=[]
                        best_by_process_truth=[]
                        for sub_est,sub_truth in zip(matrix_criteria_est,matrix_criteria_truth):
                            if len(sub_truth)>=i+1:
                                best_by_process_est.append(sub_est[i])
                                best_by_process_truth.append(sub_truth[i])
                        current_best.append(best_by_process_truth[best_by_process_est.index(max(best_by_process_est))]) 

                    return current_best

        def get_current_min(matrix_truth,matrix_criteria):

                max_long=max([len(i) for i in matrix_truth])
                current_min=[]
                for i in range(max_long):
                    # Minimo global para normalizar despues
                    current_truht_mins,current_criteria_mins=[],[]
                    for sub_truth,sub_criteria in zip(matrix_truth,matrix_criteria):
                        current_truht_mins.append(min(sub_truth[:i+1]))
                        current_criteria_mins.append(min(sub_criteria[:i+1]))
                    current_min.append(min(current_truht_mins+current_criteria_mins))

                return current_min

        def resource_allocation(listas):

            # 1. rango global
            all_data = [x for lista in listas for sub in lista for x in sub]
            min_val = min(all_data)
            max_val = max(all_data)

            # bins no uniformes
            bins = np.array([min_val,min_val + 0.25 * (max_val - min_val),min_val + 0.75 * (max_val - min_val),max_val])

            # 2. histogramas
            histogramas = []
            for lista_de_listas in listas:
                datos = [x for sub in lista_de_listas for x in sub]
                counts, _ = np.histogram(datos, bins=bins)
                histogramas.append(counts)

            # 3. obtener diferencias de histogramas
            low1= histogramas[1][0]/(histogramas[1][0]+histogramas[1][1]+histogramas[1][2])
            middle1= histogramas[1][1]/(histogramas[1][0]+histogramas[1][1]+histogramas[1][2])
            high1= histogramas[1][2]/(histogramas[1][0]+histogramas[1][1]+histogramas[1][2])
            low2= histogramas[0][0]/(histogramas[0][0]+histogramas[0][1]+histogramas[0][2])
            middle2= histogramas[0][1]/(histogramas[0][0]+histogramas[0][1]+histogramas[0][2])
            high2= histogramas[0][2]/(histogramas[0][0]+histogramas[0][1]+histogramas[0][2])


            diff_list=[low1-low2,middle1-middle2,high1-high2]

            return diff_list

            
        fig, axs = plt.subplots(2,len(self.all_packs), figsize=(2*len(self.all_packs),2*2),sharey='row',sharex='col')
        plt.subplots_adjust(top=0.9,bottom=0.2,left=0.075,right=0.99, hspace=0.,wspace=0.)

        for ax in axs.flat:
            ax.grid(True, which='major', axis='both', color='gray', linestyle='--', alpha=0.3)
        

        for i,pack in enumerate(self.all_packs):

            # Almacenar datos
            acc_p25,acc_median,acc_p75=[],[],[]
            acc_all=[]
            ra_low,ra_middle,ra_high=[],[],[]
            for test_freq in list_test_freq:

                # Obtener evolucion de accuracy
                matrix_best_est,matrix_best_truth=successive_halving(self.data_path+'/',pack)
                matrix_test_est,matrix_test_truth=successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep,test_freq])

                truth_curve=accuracy_curve( matrix_best_est,matrix_best_truth)
                test_curve=accuracy_curve( matrix_test_est,matrix_test_truth)
                current_min=get_current_min(matrix_best_truth,matrix_test_truth)
                acc_evol=accuracy_curve( matrix_test_est,matrix_test_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=test_curve,current_min=current_min)

                acc_p25.append(np.percentile(acc_evol,25))
                acc_median.append(np.median(acc_evol))
                acc_p75.append(np.percentile(acc_evol,75))
                acc_all.append(acc_evol)


                # Obtener resource allocation
                diff_list=resource_allocation([matrix_best_truth,matrix_test_truth])

                ra_low.append(diff_list[0])
                ra_middle.append(diff_list[1])
                ra_high.append(diff_list[2])

            # Dibujar graficas por entorno
            def plot_eff(ax,x,y1,y2,y3,pack,first_pack=False):
                ax.plot(x, y1, color="#e65858", label='low',marker=7,linewidth=1.5)
                ax.plot(x, y2, color="#815513", label='middle',linewidth=1.5)
                ax.plot(x, y3, color='#3cb371', label='high',marker=6,linewidth=1.5)
                ax.axhline(y=0, color='black', linestyle='--',linewidth=1)
                ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=14)
                ax.tick_params(axis='y', labelsize=12)
                ax.set_ylim(-0.25,0.4)


                if first_pack:
                    ax.set_ylabel('Proportion\ndifference in RA',fontsize=12)

            def plot_acc(ax,x,listas,first_pack=False,middle_pack=False):


                for x_i, data in zip(x, listas):

                    # puntos individuales
                    jitter = np.random.uniform(-0.015, 0.015, len(data))
                    ax.scatter(
                        np.full(len(data), x_i) + jitter,
                        data,
                        color='black',
                        s=0.5,
                        zorder=5
                    )

                    # estadísticos
                    p25 = np.percentile(data, 25)
                    med = np.median(data)
                    p75 = np.percentile(data, 75)

                    # banda IQR
                    ax.fill_between(
                        [x_i - 0.015, x_i + 0.015],
                        [p25, p25],
                        [p75, p75],
                        color='red',
                        alpha=0.5,
                        zorder=2
                    )

                    # mediana
                    ax.hlines(
                        med,
                        x_i - 0.015,
                        x_i + 0.015,
                        color='red',
                        linewidth=2,
                        zorder=6
                    )




                ax.set_ylim(-0.1,1.1)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_yticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'])
                ax.set_xlim(0,0.55)
                ax.set_xticks([0.05, 0.2, 0.35, 0.5])
                ax.set_xticklabels(['$0.05$', '$0.2$', '$0.35$', '$0.5$'])
                if first_pack:
                    ax.set_ylabel('Accuracy in\nbest-policy tracking',fontsize=12)
                if middle_pack:
                    ax.set_xlabel(r'$\varphi_c$',fontsize=12)
            
                ax.tick_params(axis='both', labelsize=12)


            first_pack=[True if i==0 else False][0]
            middle_pack=[True if i==len(self.all_packs)//2 else False][0]
            #---- curvas eff: ox=list_test_freq, oy=ra_low,ra_middle,ra_high
            plot_eff(axs[0,i],list_test_freq,ra_low,ra_middle,ra_high,pack,first_pack=first_pack)
            #---- curvas con CI acc: ox=list_test_freq, oy=[acc_p25,acc_median,acc_p75]
            plot_acc(axs[1,i],list_test_freq,acc_all,first_pack=first_pack,middle_pack=middle_pack)

        # Leyenda
        legend_elements = [
            Line2D([0], [0], marker=6, color='#3cb371',
                markerfacecolor='#3cb371', markersize=10,lw=2,
                label='high reward evaluations'),
            
            Line2D([0], [0], marker=7, color="#e65858",
                markerfacecolor="#e65858", markersize=10, lw=2,
                label='low reward evaluations'),
            
            Line2D([0], [0], color="#815513", lw=2,
                label='middle reward evaluations'),

            Line2D([0], [0], color='red', alpha=0.5, lw=7,
           label='interquartile range'),
    
            Line2D([0], [0], color='red', lw=2,
                label='median')
        ]

        axs[1,0].legend(handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(3.5, -0.3),
                ncol=5,
                frameon=False,fontsize=12)
                
        plt.savefig(self.figure_path+'/appendix/early_stopping_tradeoff.png')

