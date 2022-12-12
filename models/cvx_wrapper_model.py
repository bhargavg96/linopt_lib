__author__ = "bganguly@purdue.edu"

import numpy as np
import copy

import cvxpy as cp


class CvxBenchmarkSolverUtilityClass:
    def __init__(self,
                 num_opt_vars = None,
                 orig_cost_coeff_array = None,
                 orig_constraint_coeff_array = None,
                 constraint_rhs_array = None,
                 minimization_flag = None,
                 *args, **kwargs):

        self.num_opt_vars = copy.deepcopy(num_opt_vars)
        self.opt_cvx_variable = cp.Variable(num_opt_vars)

        self.cost_coeff_array = copy.deepcopy(orig_cost_coeff_array)

        self.constraint_coeff_array = copy.deepcopy(orig_constraint_coeff_array)

        self.constraint_rhs_array = copy.deepcopy(constraint_rhs_array.reshape((constraint_rhs_array.shape[0],)))

        self.minimization_flag = copy.deepcopy(minimization_flag)


    def run_solver(self):


        self.curr_lp_objective = cp.Minimize(self.cost_coeff_array @ self.opt_cvx_variable) if self.minimization_flag \
            else cp.Maximize(self.cost_coeff_array @ self.opt_cvx_variable)


        self.curr_lp_cvxpy_problem = cp.Problem(self.curr_lp_objective,
                                           [self.constraint_coeff_array @ self.opt_cvx_variable <= self.constraint_rhs_array,
                                            self.opt_cvx_variable >= np.zeros((self.num_opt_vars), dtype = np.float64)])

        self.curr_lp_cvxpy_problem.solve(verbose = False)

        self.final_results_dict = copy.deepcopy(self.get_results_dict())

        return self.final_results_dict


    def get_results_dict(self):

        results_dict = {}

        if self.curr_lp_cvxpy_problem.status == 'infeasible':
            results_dict['feasible'] = False
            return results_dict

        else:
            results_dict['feasible'] = True

        if self.curr_lp_cvxpy_problem.status == 'unbounded':
            results_dict['bounded'] = False
            return results_dict

        else:
            results_dict['bounded'] = True

        solution_dict = {}

        solution_dict['objective'] = self.curr_lp_cvxpy_problem.value

        for i in range(self.num_opt_vars):
            solution_dict["x_{}".format(i)] = self.opt_cvx_variable.value[i]

        results_dict['solution_dict'] = copy.deepcopy(solution_dict)

        return results_dict







