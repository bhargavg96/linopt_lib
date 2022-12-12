import pandas as pd
import numpy as np

import copy

import pprint

from models import cvx_wrapper_model, simplex_model

from lp_problem_data import *



lp_test_problems_dict = {'simplex' : {'prob4' : Problem_4_simplex_dict, 'prob19' : Problem_19_simplex_dict,
                                      'bjs36' : BJS_3_6_simplex_problem_dict, 'bjs37' : BJS_3_7_simplex_bm_problem_dict,
                                      'bjs44' : BJS_4_4_simplex_problem_dict, 'bjs45' : BJS_4_5_simplex_problem_dict,
                                      'bjs411' : BJS_4_11_simplex_problem_dict},
                         'benchmark' : {'prob4' : Problem_4_bm_dict, 'prob19' : Problem_19_bm_dict,
                                        'bjs36' : BJS_3_6_bm_problem_dict, 'bjs37' : BJS_3_7_simplex_bm_problem_dict,
                                        'bjs44' : BJS_4_4_bm_problem_dict, 'bjs45' : BJS_4_5_bm_problem_dict,
                                        'bjs411' : BJS_4_11_bm_problem_dict}}


def _main(lp_data_bm_dictionary = {},
          lp_data_simplex_dictionary = {},
          *args, **kwargs):
    pprint.pprint("The lp problem dict is : {} \n".format(lp_data_simplex_dictionary))

    pprint.pprint("First Using Commercial Benchmark Solver on Given Problem")

    cvx_solver_util = cvx_wrapper_model.CvxBenchmarkSolverUtilityClass(num_opt_vars = lp_data_bm_dictionary['num_opt_vars'],
                                                                       orig_cost_coeff_array=lp_data_bm_dictionary['cost_coeff_array'],
                                                                       orig_constraint_coeff_array=lp_data_bm_dictionary['constraint_coeff_array'],
                                                                       constraint_rhs_array= lp_data_bm_dictionary['constraint_rhs_array'],
                                                                       minimization_flag= lp_data_bm_dictionary['minimization_flag'],
                                                                       is_std_flag = lp_data_bm_dictionary['is_std_flag'],
                                                                       slack_var_sign_array= lp_data_bm_dictionary['slack_var_sign_array'])

    benchmark_solver_final_solution_dict = cvx_solver_util.run_solver()

    pprint.pprint("Commercial Benchmark Solver Results Dict is : \n")

    pprint.pprint("{} \n".format(benchmark_solver_final_solution_dict))

    simplex_solver_util = simplex_model.SimplexLPSolver(num_opt_vars = lp_data_simplex_dictionary['num_opt_vars'],
                                                        orig_cost_coeff_array=lp_data_simplex_dictionary['cost_coeff_array'],
                                                        orig_constraint_coeff_array=lp_data_simplex_dictionary['constraint_coeff_array'],
                                                        constraint_rhs_array= lp_data_simplex_dictionary['constraint_rhs_array'],
                                                        minimization_flag= lp_data_simplex_dictionary['minimization_flag'],
                                                        is_std_flag = lp_data_simplex_dictionary['is_std_flag'],
                                                        slack_var_sign_array= lp_data_simplex_dictionary['slack_var_sign_array'])

    simplex_solver_final_solution_dict = simplex_solver_util.run_solver()

    pprint.pprint("Simplex Solver Results dict is : \n")

    pprint.pprint("{} \n".format(simplex_solver_final_solution_dict))


    if benchmark_solver_final_solution_dict['feasible'] and benchmark_solver_final_solution_dict['bounded']:
        pprint.pprint("------------Preparing final results diff analytics-----------------")

        for i in range(lp_data_simplex_dictionary['num_opt_vars']):
            print("x_{}, benchmark: {}, simplex: {}".format(i, benchmark_solver_final_solution_dict['solution_dict']["x_{}".format(i)],
                                                            simplex_solver_final_solution_dict['solution_dict']["x_{}".format(i)]))

        print("objective, benchmark: {}, simplex: {}".format(benchmark_solver_final_solution_dict['solution_dict']['objective'],
                                                             simplex_solver_final_solution_dict['solution_dict']['objective']))

if __name__ == '__main__':
    problem_name = 'bjs411'
    lp_data_bm_dict = copy.deepcopy(lp_test_problems_dict['benchmark'][problem_name])

    lp_data_simplex_dict = copy.deepcopy(lp_test_problems_dict['simplex'][problem_name])

    _main(lp_data_bm_dictionary=lp_data_bm_dict,
          lp_data_simplex_dictionary= lp_data_simplex_dict)