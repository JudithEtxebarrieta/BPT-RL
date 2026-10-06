from final.src.analysis_data import *

def df_analysis_per_env(seeds,env,path):

    # STEP 1: Collect data for the graphs by seed.
    pack = 'pack_PPO_'+env
    costs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    optimal_train_n_ep, optimal_test_cost, optimal_test_n_ep = 50, 0.2, 5
    default_train_n_ep, default_test_n_ep, default_test_freq = 100, 5
    all_default_test_freq={ # Environments included in the empirical analysis of the paper.
                            'LunarLanderContinuous':10,'BipedalWalker': 5,
                            'Ant': 5,'HalfCheetah': 20,'Hopper': 20,'Swimmer': 10,'Walker2d': 20,
                            # Additional environment as an example of reproducibility.
                            'Pendulum': 20}
    
    default_test_freq=all_default_test_freq[env]

    for seed in seeds:

        print(f"Comenzando seed {seed}")

        datagenerator = DataGenerator(pack, [seed],path)
        datagenerator.add_all_criteria_eff(optimal_train_n_ep,optimal_test_cost,optimal_test_n_ep,single_seed=seed)

        for cost in tqdm(costs):
            datagenerator.add_test_cost_truth(cost,default_test_n_ep,single_seed=seed)
            datagenerator.add_all_criteria_estimates(default_train_n_ep,default_test_n_ep,default_test_freq,optimal_train_n_ep,cost,optimal_test_n_ep,single_seed=seed)

        print(f"Terminada seed {seed}")

    # STEP 2: Joint the data from all seeds.
    DataConverter.join_df_analysis_seed(path+'/SB3_PPO_'+env+'/analysis_data/',seeds,only_test_with_cost_truth=False)
    DataConverter.join_df_analysis_seed(path+'/SB3_PPO_'+env+'/analysis_data/',seeds,only_test_with_cost_truth=True)
    DataConverter.join_df_analysis_seed(path+'/SB3_PPO_'+env+'/analysis_data/',seeds,only_default_est=True)

    # STEP 3: Additional data that does not need to be joined (it is already generated as a single set).
    PackProcessDataGenerator.df_analysis_all_seq_est(seeds,env,path)
    PackProcessDataGenerator.df_analysis_all_seq_val_cost_len(seeds,env,path)


    

# Example 
df_analysis_per_env([1,2,3,4],'Pendulum','final/results/data')