
#==================================================================================================
# Dependencies
#==================================================================================================
from reproducibility.src.analysis_data import *
from reproducibility.src.grapher import *
import os
import zipfile

#==================================================================================================
# Auxiliary functions
#==================================================================================================

def add_p25prec_by_conf_and_regions_per_pack(pack,prec_metric='relative_perc_criteria_best'):
        
        '''
        Store the 25th percentiles of accuracy by region and criterion configuration, in order to
        subsequently calculate the optimal global configuration, taking all setups into account.
        '''

        data_path=data_common_path+'/'+pack.replace('pack','SB3')+'/'
        
        # Train data.
        prec1,prec2,prec3,conf_list=from_df_data_to_graph_data(
            [data_path+'learning_regions.csv',data_path+'df_train_prec.csv'],pack,which_graph='train_conf_prec',
            prec_metric=prec_metric
        )

        if not os.path.exists(data_common_path+'/train_pack_conf_p25prec.csv'):
            df = pd.DataFrame(columns=['pack_conf','p25_initialization','p25_learning','p25_stabilization'])
            os.makedirs(os.path.dirname(data_common_path+'/train_pack_conf_p25prec.csv'), exist_ok=True)
            df.to_csv(data_common_path+'/train_pack_conf_p25prec.csv', index=False)
        else:
            df=pd.read_csv(data_common_path+'/train_pack_conf_p25prec.csv')

        if not df['pack_conf'].str.contains(pack.replace('pack_',''), na=False).any():
            for conf,i,j,k in zip(conf_list,prec1,prec2,prec3):
                df.loc[len(df)]=['SB3'+pack.replace('pack','')+'_'+str(conf),np.percentile(i, 25),np.percentile(j, 25),np.percentile(k, 25)]
                df.to_csv(data_common_path+'/train_pack_conf_p25prec.csv',index=False)

        # Test data.
        prec1,prec2,prec3,n_ep_list,freq_list=from_df_data_to_graph_data(
            [data_path+'learning_regions.csv',data_path+'df_test_prec.csv'],pack,which_graph='test_conf_prec_cost',
            prec_metric=prec_metric,n_ep_type='with_cost_freq'
        )

        if not os.path.exists(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv'):
            df = pd.DataFrame(columns=['pack_conf','p25_initialization','p25_learning','p25_stabilization'])
            os.makedirs(os.path.dirname(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv'), exist_ok=True)
            df.to_csv(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv', index=False)
        else:
            df=pd.read_csv(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv')

        if not df['pack_conf'].str.contains(pack.replace('pack_',''), na=False).any():
            for idx_cost in range(len(prec1)):
                for freq,i,j,k in zip(freq_list,prec1[idx_cost],prec2[idx_cost],prec3[idx_cost]):
                    df.loc[len(df)]=['SB3'+pack.replace('pack','')+'_'+str(n_ep_list[idx_cost])+'_'+str(freq),np.percentile(i, 25),np.percentile(j, 25),np.percentile(k, 25)]
                    df.to_csv(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv',index=False)

        # Initialize the database that will store the configurations.
        if not os.path.exists(data_common_path+'/configurations.csv'):
                df = pd.DataFrame(columns=['pack','train_default','train_opt','test_default','test_cost_freq_opt'])
                os.makedirs(os.path.dirname(data_common_path+'/configurations.csv'), exist_ok=True)
                df.to_csv(data_common_path+'/configurations.csv', index=False)
        else:
            df=pd.read_csv(data_common_path+'/configurations.csv')

def from_p25prec_to_common_optimal_conf():

    '''
    Based on the 25th percentile accuracy by region and criterion configuration, save the best overall
    configuration for all environments, which will be the one with the highest 25th percentile accuracy.
    '''

    # Reading databases.
    train_p5=pd.read_csv(data_common_path+'/train_pack_conf_p25prec.csv')
    test_freq_p5=pd.read_csv(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv')
    df_conf=pd.read_csv(data_common_path+'/configurations.csv')

    all_row=['all']+[None]*(df_conf.shape[1]-1) # Initialize the df_conf row associated with the general configurations.
    
    
    # Optimal global train configuration.
    list_conf = list(set(train_p5['pack_conf'].str.split('_').str[-1].tolist()))
    list_sum_p5=[]
    for conf in list_conf:
        df = train_p5[train_p5['pack_conf'].str.endswith('_'+conf, na=False)]
        list_sum_p5.append(df.iloc[:, -3:].sum().sum())
        
    idx_max = list_sum_p5.index(max(list_sum_p5))
    all_row[2]=list_conf[idx_max]

    # Optimal global test configuration.
    list_conf = list(set(test_freq_p5['pack_conf'].str.split('_').str[-2:].str.join('_').tolist()))
    list_sum_p5=[]
    for conf in list_conf:
        df = test_freq_p5[test_freq_p5['pack_conf'].str.endswith('_'+conf, na=False)]
        list_sum_p5.append(df.iloc[:, -3:].sum().sum())
        
    idx_max = list_sum_p5.index(max(list_sum_p5))
    all_row[4]=list_conf[idx_max]

    
    # Add global optimal configurations to the database of configurations, and initialize the remaining rows with the name of the setup.
    df_conf.loc[len(df_conf)] = all_row
    setups = list(set(df["pack_conf"].str.split("_").str[:3].str.join("_").tolist()))
    for setup in setups:
        df_conf.loc[len(df_conf)] = [setup.replace('SB3','pack')]+[None]*(df_conf.shape[1]-1)

    df_conf.to_csv(data_common_path+'/configurations.csv',index=False)

def from_p25prec_to_pack_optimal_conf(env,default_test_freq):

    '''
    Based on the 25th percentile accuracy by region and criteria configuration, it saves the best
    overall configuration for the given environment, which will be the one with the highest 25th 
    percentile accuracy in that environment.
    '''

    # Read databases and reduce them to the relevant rows.
    train_p25=pd.read_csv(data_common_path+'/train_pack_conf_p25prec.csv')
    train_p25=train_p25[train_p25["pack_conf"].str.contains(env, na=False)]
    test_freq_p25=pd.read_csv(data_common_path+'/test_cost_freq_pack_conf_p25prec.csv')
    test_freq_p25=test_freq_p25[test_freq_p25["pack_conf"].str.contains(env, na=False)]
    df_conf=pd.read_csv(data_common_path+'/configurations.csv')

    all_row=[None]*(df_conf.shape[1]-1) # Initialize the df_conf row associated with the general configurations.
    
    
    # Optimal train configuration.
    list_conf = list(set(train_p25['pack_conf'].str.split('_').str[-1].tolist()))
    list_sum_p25=[]
    for conf in list_conf:
        df = train_p25[train_p25['pack_conf'].str.endswith('_'+conf, na=False)]
        list_sum_p25.append(df.iloc[:, -3:].sum().sum())
        
    idx_max = list_sum_p25.index(max(list_sum_p25))
    all_row[1]=list_conf[idx_max]

    # Default configuration of train.
    all_row[0]=100

    # Optimal test configuration.
    list_conf = list(set(test_freq_p25['pack_conf'].str.split('_').str[-2:].str.join('_').tolist()))
    list_sum_p5=[]
    for conf in list_conf:
        df = test_freq_p25[test_freq_p25['pack_conf'].str.endswith('_'+conf, na=False)]
        list_sum_p5.append(df.iloc[:, -3:].sum().sum())
        
    idx_max = list_sum_p5.index(max(list_sum_p5))
    all_row[3]=list_conf[idx_max]

    # Default configuration of test.
    all_row[2]='5_'+str(default_test_freq)


    # Add the optimal configurations of the pack to the settings database.
    df_conf = df_conf.astype(object)
    df_conf.loc[df_conf["pack"].str.contains(env, na=False), df_conf.columns.drop("pack")] = all_row

    df_conf.to_csv(data_common_path+'/configurations.csv',index=False)

#==================================================================================================
# Main classes
#==================================================================================================

class UnzipAnalysisData():

    '''
    Unzip the all_data.zip folders provided for each algorithm-environment pair considered
    in the experimental analysis of the article. Each of these folders contains all the databases 
    on which the analysis is based, which can be reproduced using the `generate_analysis_data.py` script,
    which in turn is based on the data stored after running `generate_process_data.py`.'''

    def __init__(self, all_packs=[
                                'SB3_PPO_BipedalWalker',
                                'SB3_PPO_LunarLanderContinuous',
                                'SB3_PPO_Swimmer',
                                'SB3_PPO_Ant',
                                'SB3_PPO_Walker2d',
                                'SB3_PPO_HalfCheetah',
                                'SB3_PPO_Hopper'
                                ]):

        for pack in all_packs:
            pack_path = data_common_path + '/' + pack
            archive_path = pack_path + '/all_data.zip'
            analysis_path = pack_path + '/analysis_data'

            if os.path.isfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(pack_path + '/')
            elif not os.path.isdir(analysis_path):
                raise FileNotFoundError(
                    f"Neither {archive_path} nor {analysis_path} is available. "
                    "Generate the analysis data for all seven paper environments first."
                )
    
class ComputeCommonData():
    '''Save the best configuration of the criteria, taking all setups into account.'''

    def __init__(self,
                 all_packs=[
                            'pack_PPO_BipedalWalker',
                            'pack_PPO_LunarLanderContinuous',
                            'pack_PPO_Swimmer',
                            'pack_PPO_Ant',
                            'pack_PPO_Walker2d',
                            'pack_PPO_HalfCheetah',
                            'pack_PPO_Hopper'
                            ],
                 default_test_freq=[5,10,10,5,20,20,20]
                 ):

        # Store 25 precision values for each region for every possible configuration per pack.
        for pack in all_packs:
            add_p25prec_by_conf_and_regions_per_pack(pack)

        # Get and save the configuration with the highest p25 precision for the average across all regions and packs.
        from_p25prec_to_common_optimal_conf()

        # Complete the configuration details for each pack.
        for i,pack in enumerate(all_packs):
            from_p25prec_to_pack_optimal_conf(pack.split('_')[2],default_test_freq[i])


# Main programme.
# NEW: run from the repository root with the activated project environment.
data_common_path='reproducibility/results/data'
UnzipAnalysisData()
ComputeCommonData()
