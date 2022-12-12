__author__ = "bganguly@purdue.edu"

import numpy as np
import copy

import pprint


class SimplexLPSolver:
    def __init__(self,
                 num_opt_vars = None,
                 orig_cost_coeff_array = None,
                 orig_constraint_coeff_array = None,
                 constraint_rhs_array = None,
                 slack_var_sign_array = None,
                 minimization_flag = None,
                 is_std_flag = False,
                 error_tolerance = 1e-6,
                 *args, **kwargs):

        self.orig_num_opt_vars = copy.deepcopy(num_opt_vars)
        self.orig_cost_coeff_array = copy.deepcopy(np.array(orig_cost_coeff_array,
                                                 dtype = np.float64))
        self.orig_constraint_coeff_array = copy.deepcopy(np.array(orig_constraint_coeff_array,
                                                             dtype = np.float64))
        self.constraint_rhs_array = copy.deepcopy(np.array(constraint_rhs_array,
                                                 dtype = np.float64))

        self.num_slack_vars = len(slack_var_sign_array)

        self.slack_var_sign_array = copy.deepcopy(np.array(slack_var_sign_array,
                                                           dtype = np.float64))

        self.minimization_flag = copy.deepcopy(minimization_flag)

        self.results_dict = {}

        self.is_std_flag = copy.deepcopy(is_std_flag)

        self.total_std_vars = self.orig_num_opt_vars + self.num_slack_vars

        self.error_tolerance = copy.deepcopy(error_tolerance)

        self.bounded_problem_flag = True

        self.convergence_reached_flag = False

        self.num_constraints = copy.deepcopy(self.num_slack_vars)


    def lp_std_form_preprocessor(self):
        """
        This subroutine converts the given LP to standard form.
        - This involves conversion to minimization problem.
        - Introduction of slack variables with appropriate sign.
        """

        print("LP standardization pre-processor starts... ")
        if self.is_std_flag:
            return

        self.orig_cost_coeff_array = self.orig_cost_coeff_array if self.minimization_flag else -1.*self.orig_cost_coeff_array

        self.slack_cost_coeff_array = np.zeros((1,self.num_slack_vars),
                                               dtype = np.float64)


        self.std_cost_coeff_array = np.concatenate((self.orig_cost_coeff_array,
                                                    self.slack_cost_coeff_array),
                                                  axis = 1)

        pprint.pprint("LP standardization: Merged slack 0 costs with original cost coeffs.")

        self.slack_constraint_coeff_array = np.eye(self.num_slack_vars,
                                                   dtype = np.float64)*self.slack_var_sign_array

        pprint.pprint("LP standardization: Initialized slack constraint coeffs with appropriate sign.")

        self.std_constraint_coeff_array = np.concatenate((self.orig_constraint_coeff_array,
                                                          self.slack_constraint_coeff_array),
                                                         axis = 1)

        pprint.pprint("LP standardization: Merged slack constraint coeffs with original constraint coeffs.")

        print("The given LP is now standardized.")




    def build_phase1_tableau_array(self):

        pprint.pprint("Building Phase 1 tableau array...")

        self.num_artificial_vars = copy.deepcopy(self.num_slack_vars)

        self.artificial_vars_const_coeff_array = np.eye(self.num_slack_vars,
                                                  dtype = np.float64)

        pprint.pprint("Phase 1 Build: Initialized artificial variables with appropriate constraint coeffs.")



        self.phase_1_constraint_coeff_array = np.concatenate((self.std_constraint_coeff_array,
                                                              self.artificial_vars_const_coeff_array),
                                                             axis = 1)

        pprint.pprint("Phase 1 Build: Merged constraint coeffs of std vars and artificial vars.")

        self.phase1_tableau_array_top_cost_row = np.concatenate((np.sum(self.std_constraint_coeff_array,
                                                           axis = 0),
                                                    np.zeros(self.num_artificial_vars, ),
                                                    np.array([np.sum(self.constraint_rhs_array)]))).reshape(1, self.total_std_vars + self.num_artificial_vars +1)

        pprint.pprint("Phase 1 Build: Prepared top row of tableau array from cost coeffs and initial RHS.")

        self.phase1_tableau_array_constraint_rhs_rows = np.concatenate((self.phase_1_constraint_coeff_array,
                                                                        self.constraint_rhs_array),
                                                                       axis = 1)

        self.tableau_array = np.concatenate((self.phase1_tableau_array_top_cost_row,
                                                    self.phase1_tableau_array_constraint_rhs_rows),
                                                   axis = 0)

        pprint.pprint("Phase 1 Build: Constraint RHS, constraint coeffs and top cost row combined. Phase 1 tableau initialization complete.")


        self.basic_var_assignment_dict = {i+1 : self.total_std_vars + i for i in range(self.num_artificial_vars)}

        self.var_category_id_dict = {'orig_vars' : range(self.orig_num_opt_vars),
                                     'slack_vars' : range(self.orig_num_opt_vars, self.total_std_vars),
                                     'artificial_vars' : range(self.total_std_vars, self.total_std_vars + self.num_artificial_vars)}

        #self.run_tableau_row_operations()


        pprint.pprint("Following are the different optimization var indexes and their categories: {}".format(self.var_category_id_dict))



    def build_phase2_tableau_array(self):

        print("At the end of phase 1 the basic vars are: {}".format(self.basic_var_assignment_dict))

        print("Also, the artificial var ids are: " + str(['x_{}'.format(i) for i in self.var_category_id_dict['artificial_vars']]))

        redundant_row_list = []

        redundant_vars_list = []

        for k in self.basic_var_assignment_dict.keys():
            if self.basic_var_assignment_dict[k] in self.var_category_id_dict['artificial_vars']:

                print("row: {}, artificial var: x_{} being tested for redundancy..".format(k,
                                                                                           self.basic_var_assignment_dict[k]))
                redundant_constraints_flag = True
                for j in range(self.total_std_vars):
                    if (np.abs(self.tableau_array[k][j]) > self.error_tolerance) and (j not in self.basic_var_assignment_dict.values()):
                        self.basic_var_assignment_dict[k] = j
                        pprint.pprint("{} row is now assigned to std var x_{}, redundancy is now removed.".format(k,j))
                        redundant_constraints_flag = False
                        break

                if redundant_constraints_flag:
                    redundant_vars_list.append(self.basic_var_assignment_dict[k])
                    redundant_row_list.append(k)

        print("redundant constraint rows are: {}".format(self.tableau_array[redundant_row_list]))

        print("redundant vars are: {}".format(redundant_vars_list))

        phase2_tableau_top_row = np.concatenate((-self.std_cost_coeff_array, np.array([[0.]])),
                                                axis = 1)

        phase2_tableau_array_constraint_rhs_rows = self.tableau_array[:, list(range(self.total_std_vars)) + [-1]]

        new_basic_var_assignment_dict = {}

        phase2_final_tableau_array = copy.deepcopy(phase2_tableau_top_row)

        phase2_constraint_count = 1

        for j in range(self.num_slack_vars):
            if 1+j not in redundant_row_list:
                curr_row = np.array([phase2_tableau_array_constraint_rhs_rows[1+j]], dtype = np.float64)
                phase2_final_tableau_array = np.concatenate((phase2_final_tableau_array, curr_row),
                                                            axis = 0)
                new_basic_var_assignment_dict[phase2_constraint_count] = self.basic_var_assignment_dict[1+j]
                phase2_constraint_count += 1

        self.basic_var_assignment_dict = copy.deepcopy(new_basic_var_assignment_dict)

        self.tableau_array = copy.deepcopy(phase2_final_tableau_array)

        self.num_constraints = self.tableau_array.shape[0] - 1


        pprint.pprint("Perform row operations once to reflect initial basis for Phase 2 table.")

        #This is done to preserve initial identity basis for phase 2 starting point.

        self.run_tableau_row_operations()




    def run_tableau_row_operations(self,
                                   *args,
                                   **kwargs):

        pprint.pprint("Row operations begin for current iter..")

        num_tableau_rows = self.tableau_array.shape[0]

        for i in range(1,num_tableau_rows):

            basic_var_i = self.basic_var_assignment_dict[i]
            self.tableau_array[i, :] *= 1./(self.tableau_array[i,basic_var_i])
            for j in range(0, num_tableau_rows):
                if j != i:
                    self.tableau_array[j, :] -= self.tableau_array[i, :]*self.tableau_array[j,basic_var_i]

        pprint.pprint("Row operations end for current iter.")


    def build_recession_dict(self):

        pprint.pprint("Building recession dict for the current LP problem.")
        self.recession_dict = {}

        basis_entering_variable = np.argmax(self.tableau_array[0, : -1])

        for i in range(self.num_constraints):

            self.recession_dict[self.basic_var_assignment_dict[i + 1]] = (self.tableau_array[i + 1][-1], - self.tableau_array[i + 1][basis_entering_variable])

        for j in range(self.total_std_vars):
            if j not in self.recession_dict.keys() and j != basis_entering_variable:
                self.recession_dict[j] = (0., 0.)
            elif j == basis_entering_variable:
                self.recession_dict[j] = (0.,1.)

        self.recession_dict = dict(sorted(self.recession_dict.items())) #for var x_i: (t_i1, t_i2) means x_i = t_i1 + c*t_i2

        pprint.pprint("Recession dict for the current LP problem built successfully.")






    def lexicographic_tableau_array_update_service(self):

        if np.max(self.tableau_array[0, : -1]) < 0. + self.error_tolerance: #check for convergence
            self.convergence_reached_flag = True
            return

        basis_entering_variable = np.argmax(self.tableau_array[0, : -1]) #find entering variable.

        print("current iteration entering var is x_{}".format(basis_entering_variable))

        if np.all(self.tableau_array[ 1:, basis_entering_variable] < 0 + self.error_tolerance): #check for boundedness
            self.bounded_problem_flag = False
            return

        row_ratio_list = []

        row_ratio_list.append(np.inf)

        for row in range(self.num_constraints):
            if self.tableau_array[1+row,basis_entering_variable] < 0 + self.error_tolerance:
                row_ratio_list.append(np.inf)
            else:
                row_ratio_list.append(self.tableau_array[1+row,-1]/self.tableau_array[1+row,basis_entering_variable])

        min_row_ids = list(np.where(row_ratio_list == np.min(row_ratio_list))[0])

        for col in range(0,self.tableau_array.shape[1]-1): #column wise lexicographic sorting
            if len(min_row_ids) == 1:
                break

            curr_col_ratio_dict = {i : self.tableau_array[i, col]/self.tableau_array[i, basis_entering_variable] \
                                   for i in min_row_ids}

            new_min_row_ids = []

            min_col_ratio_val = np.min(list(curr_col_ratio_dict.values()))

            for k in curr_col_ratio_dict.keys():
                if curr_col_ratio_dict[k] == min_col_ratio_val:
                    new_min_row_ids.append(k)

            min_row_ids = copy.deepcopy(new_min_row_ids)

        leaving_var_row_id = np.int64(min_row_ids[0]) #pick the lexicographically smallest var.

        print("leaving variable is : x_{}".format(self.basic_var_assignment_dict[leaving_var_row_id]))

        self.basic_var_assignment_dict[leaving_var_row_id] = copy.deepcopy(basis_entering_variable) #change entering-leaving assignment for given row.


        self.run_tableau_row_operations()


    def get_current_iter_results_dict(self):
        """
        This is a util function to create a summary dict of results.
        """

        curr_iter_results_dict = {}

        curr_iter_results_dict['objective'] = self.tableau_array[0][-1] if self.minimization_flag else self.tableau_array[0][-1] * (-1.)

        for i in range(self.num_constraints):

            if self.basic_var_assignment_dict[1+i] in range(self.orig_num_opt_vars):

                curr_iter_results_dict["x_{}".format(self.basic_var_assignment_dict[1+i])] = self.tableau_array[i+1][-1]

        for j in range(self.orig_num_opt_vars):

            if j not in self.basic_var_assignment_dict.values():

                curr_iter_results_dict["x_{}".format(j)] = 0.

        return curr_iter_results_dict


    def run_solver(self):

        pprint.pprint("---------------------Simplex Model Run begins----------------------")

        self.final_results_dict = {}

        self.lp_std_form_preprocessor()

        self.build_phase1_tableau_array()

        tableau_run_iter_count = 1

        while not self.convergence_reached_flag:

            self.lexicographic_tableau_array_update_service()

            pprint.pprint("Iteration: {}, Optimization Variables and Objective: {}".format(tableau_run_iter_count,
                                                                     self.get_current_iter_results_dict()))
            tableau_run_iter_count += 1


        print("Simplex Phase 1 completed.")


        if self.tableau_array[0, -1] > self.error_tolerance:

            print("Problem is infeasible")

            self.final_results_dict['feasible'] = False

            return self.final_results_dict


        self.final_results_dict['feasible'] = True

        self.build_phase2_tableau_array()


        self.convergence_reached_flag = False


        while (not self.convergence_reached_flag) and self.bounded_problem_flag:

            self.lexicographic_tableau_array_update_service()

            pprint.pprint("Iteration: {}, Optimization Variables and Objective: {}".format(tableau_run_iter_count,
                                                                     self.get_current_iter_results_dict()))
            tableau_run_iter_count += 1

        if not self.bounded_problem_flag:

            print("Problem is feasible although unbounded")
            self.build_recession_dict()

            self.final_results_dict['bounded'] = False

            self.final_results_dict['recession_dict'] = self.recession_dict

        else:
            print("Problem is both feasible and bounded")

            self.final_results_dict['bounded'] = True

            self.final_results_dict['solution_dict'] = self.get_current_iter_results_dict()


        return self.final_results_dict





