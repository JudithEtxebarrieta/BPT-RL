# LEGACY MAIN (replaced by the configurable entry point below):
# from final.src.grapher import *
#
# data_path='final/results/data'
# figure_path='final/results/figures'
#
# grapher=MainGrapher(data_path,figure_path)
# grapher.deg_prec_eff()
# grapher.patterns_deg_acc_eff()
# grapher.criteria_comparison()
# grapher.criteria_comparison(comparison_type='in_which_deg_best')
#
# grapher=AppendixGrapher(data_path,figure_path)
# grapher.estimations()
# grapher.estimations(train_white=True)
# grapher.estimations_selected_conf()
# grapher.criteria_configurations()
# grapher.cost_of_cost_driven_test()
# grapher.val_cost_throughout_sequence()
# grapher.val_freq_throughout_sequence()
# grapher.test_default_vs_with_cost()
# grapher.regions_with_deg()
# grapher.criteria_consequences(consequence_type='early_stopping_all')
# grapher.criteria_consequences(consequence_type='early_stopping_eff')
# grapher.criteria_consequences(consequence_type='early_stopping_prec')
# grapher.resource_allocation()

# NEW: generate main-analysis and appendix figures from configurable paths.
import argparse
import os

from reproducibility.src.grapher import MainGrapher, AppendixGrapher


def main():
    parser = argparse.ArgumentParser(description='Generate the reproducibility analysis figures.')
    parser.add_argument('--data_path', default='reproducibility/results/data')
    parser.add_argument('--figure_path', default='reproducibility/results/figures')
    args = parser.parse_args()

    os.makedirs(args.figure_path, exist_ok=True)

    grapher = MainGrapher(args.data_path, args.figure_path)
    grapher.deg_prec_eff()
    grapher.patterns_deg_acc_eff()
    grapher.criteria_comparison()
    grapher.criteria_comparison(comparison_type='in_which_deg_best')

    grapher = AppendixGrapher(args.data_path, args.figure_path)
    grapher.estimations()
    grapher.estimations(train_white=True)
    grapher.estimations_selected_conf()
    grapher.criteria_configurations()
    grapher.cost_of_cost_driven_test()
    grapher.val_cost_throughout_sequence()
    grapher.val_freq_throughout_sequence()
    grapher.test_default_vs_with_cost()
    grapher.regions_with_deg()
    grapher.criteria_consequences(consequence_type='early_stopping_all')
    grapher.criteria_consequences(consequence_type='early_stopping_eff')
    grapher.criteria_consequences(consequence_type='early_stopping_prec')
    grapher.resource_allocation()


if __name__ == '__main__':
    main()
