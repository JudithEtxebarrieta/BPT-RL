from final.src.grapher import *

data_path='final/results/data'
figure_path='final/results/figures'


# Main analysis
grapher=MainGrapher(data_path,figure_path) 

grapher.deg_prec_eff()
grapher.patterns_deg_acc_eff()
grapher.criteria_comparison()
grapher.criteria_comparison(comparison_type='in_which_deg_best')

# Appendixes
grapher=AppendixGrapher(data_path,figure_path) 

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