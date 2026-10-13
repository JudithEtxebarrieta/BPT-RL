import argparse

from reproducibility.src.process_data import *

class DataGenerator():

    '''
    A main class that specifies the resources needed to generate the data used by the environments in the paper's analysis,
    and allows the same data to be generated (facilitating reproducibility).

    * `environmnet`: 'LunarLanderContinuous', 'BipedalWalker', 'Walker2d', 'Ant', 'HalfCheetah', 'Hopper', 'Swimmer'.
    * `data_path`: directory where the data will be stored.
    '''

    def __init__(self,environment, data_path):
        self.environment=environment
        self.data_path=data_path

    def check_resource_availability(self, available_CPUs):
        '''
        Information on the resources required to obtain the data used for the analysis by environment. 
        By introducing `available_CPUs`—the number of available CPUs—this function indicates whether 
        they are sufficient to generate the data and recommends using additional CPUs, if available, 
        to speed up the generation of ground truth data.

        * `available_CPUs`: available CPUs for running the learning process.
        '''

        # Packs included in the analysis of the paper.
        if self.environment=='LunarLanderContinuous':
            PackOptions.PPO_LunarLanderContinuous(None,'','',None,available_resources=available_CPUs)
        if self.environment=='BipedalWalker':
            PackOptions.PPO_BipedalWalker(None,'','',None,available_resources=available_CPUs)

        if self.environment=='Walker2d':
            PackOptions.PPO_Walker2d(None,'','',None,available_resources=available_CPUs)
        if self.environment=='Ant':
            PackOptions.PPO_Ant(None,'','',None,available_resources=available_CPUs)
        if self.environment=='HalfCheetah':
            PackOptions.PPO_HalfCheetah(None,'','',None,available_resources=available_CPUs)
        if self.environment=='Hopper':
            PackOptions.PPO_Hopper(None,'','',None,available_resources=available_CPUs)
        if self.environment=='Swimmer':
            PackOptions.PPO_Swimmer(None,'','',None,available_resources=available_CPUs)

        # Additional pack as an example of reproducibility.
        if self.environment=='Pendulum':
            PackOptions.PPO_Pendulum(None,'','',None,available_resources=available_CPUs)
        
    def default_process_data(self,seed,truth_n_workers=1):
        '''
        Run environment-seed learning processes while saving the data for the simulations.

        * `truth_n_workers`: The degree of parallelization to be considered for the interaction that 
        generates the data for estimating the ground truth. The specified number must be different 
        (either lower or higher) from the number of CPUs required to reproduce the training process,
        i.e., that recommended by the `check_resource_availability` function. It is recommended to 
        use a number of workers that is greater than (if possible) or equal to that indicated by that function.
        '''

        if __name__ == "__main__": 

            # Packs included in the analysis of the paper.
            if self.environment=='LunarLanderContinuous':
                PackOptions.PPO_LunarLanderContinuous(seed,'pack_PPO_LunarLanderContinuous_seed'+str(seed)+'_',self.data_path,truth_n_workers)
            if self.environment=='BipedalWalker':
                PackOptions.PPO_BipedalWalker(seed,'pack_PPO_BipedalWalker_seed'+str(seed)+'_',self.data_path,truth_n_workers)

            if self.environment=='Walker2d':
                PackOptions.PPO_Walker2d(seed,'pack_PPO_Walker2d_seed'+str(seed)+'_',self.data_path,truth_n_workers)
            if self.environment=='Ant':
                PackOptions.PPO_Ant(seed,'pack_PPO_Ant_seed'+str(seed)+'_',self.data_path,truth_n_workers)
            if self.environment=='HalfCheetah':
                PackOptions.PPO_HalfCheetah(seed,'pack_PPO_HalfCheetah_seed'+str(seed)+'_',self.data_path,truth_n_workers)
            if self.environment=='Hopper':
                PackOptions.PPO_Hopper(seed,'pack_PPO_Hopper_seed'+str(seed)+'_',self.data_path,truth_n_workers)
            if self.environment=='Swimmer':
                PackOptions.PPO_Swimmer(seed,'pack_PPO_Swimmer_seed'+str(seed)+'_',self.data_path,truth_n_workers)

            # Additional pack as an example of reproducibility.
            if self.environment=='Pendulum':
                PackOptions.PPO_Pendulum(seed,'pack_PPO_Pendulum_seed'+str(seed)+'_',self.data_path,truth_n_workers)


# Previous example retained for reference:
# datagenerator=DataGenerator('Pendulum','final/results/data/SB3_PPO_Pendulum')
# datagenerator.check_resource_availability(available_CPUs=8)
#
# for seed in [1,2,3,4]:
#     datagenerator.default_process_data(seed,truth_n_workers=6)


# NEW: command-line entry point for generating one environment-seed process.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and store one PPO environment-seed learning process."
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=[
            "LunarLanderContinuous", "BipedalWalker", "Walker2d", "Ant",
            "HalfCheetah", "Hopper", "Swimmer", "Pendulum",
        ],
        help="Environment name; Pendulum is an additional example outside the paper analysis.",
    )
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    parser.add_argument(
        "--path",
        required=True,
        help="Root directory for the generated data, e.g. reproducibility/results/data.",
    )
    parser.add_argument(
        "--available_CPUs",
        required=True,
        type=int,
        help="Available CPUs; used for the ground-truth interaction and resource reporting.",
    )
    args = parser.parse_args()

    generator = DataGenerator(args.env, args.path)
    generator.check_resource_availability(args.available_CPUs)
    generator.default_process_data(args.seed, truth_n_workers=args.available_CPUs)
