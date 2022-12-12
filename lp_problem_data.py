import numpy as np

import copy

def create_prob_19_slack_var_array(num_constraints = None,
                                   negative_indices = []):
    sign_list = []

    for i in range(num_constraints):
        if i in negative_indices:
            sign_list.append([-1.])
        else:
            sign_list.append([1.])

    return np.array(sign_list)

def create_prob_19_constraint_list(num_ele = 18,
                                   indices_list = [],
                                   sign = 1.,
                                   *args, **kwargs):
    answer_list = []

    for i in range(num_ele):
        if i+1 in indices_list:
            answer_list.append(1.*sign)
        else:
            answer_list.append(0.)

    return copy.deepcopy(answer_list)

Problem_4_simplex_dict = {'num_opt_vars' : 4,
                  'cost_coeff_array' : np.array([[0., 0., 0., 1.]]),
                  'constraint_coeff_array' : np.array([[9., 6., 4., 0],
                                              [5., 8., 11., 0.],
                                              [50., 75., 100., 0.],
                                              [-9., -12., -10., 5./7.],
                                              [-6., -4., -6., 2./7.]]),
                  'constraint_rhs_array' : np.array([[200.], [400.], [1850.] ,[0.] , [0.]]),
                  'minimization_flag' : False,
                  'is_std_flag' : False,
                  'slack_var_sign_array' : np.array([[1.], [1.], [1.], [1.], [1.]])}

Problem_4_bm_dict = {'num_opt_vars' : 4,
                  'cost_coeff_array' : np.array([[0., 0., 0., 1.]]),
                  'constraint_coeff_array' : np.array([[9., 6., 4., 0],
                                                       [5., 8., 11., 0.],
                                                       [50., 75., 100., 0.],
                                                       [-9., -12., -10., 5./7.],
                                                       [-6., -4., -6., 2./7.]]),
                  'constraint_rhs_array' : np.array([[200.], [400.], [1850.] ,[0.] , [0.]]),
                  'minimization_flag' : False,
                  'is_std_flag' : False,
                  'slack_var_sign_array' : np.array([[1.], [1.], [1.], [1.], [1.]])}



Problem_19_simplex_dict = {'num_opt_vars' : 18,
                      'cost_coeff_array' : np.array([[6., 6., 6.,
                                                      6.1, 6.1,
                                                      5.9, 5.9 , 5.9 , 5.9,
                                                      5.8 , 5.8 , 5.8 , 5.8,
                                                      6.8 , 6.8 , 6.8,
                                                      7.3 , 7.3]]),
                      'constraint_coeff_array' : np.array([create_prob_19_constraint_list(num_ele = 18, indices_list = [1]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [2]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [3]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [4]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [5]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [6]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [7]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [8]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [9]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [10]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [11]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [12]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [13]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [14]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [15]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [16]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [17]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [18]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [1, 2, 3]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [4, 5]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [6, 7, 8, 9]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [10, 11, 12, 13]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [14, 15, 16]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [17, 18]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [1,6,10,14]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [1,6,10,14]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [4,7,11]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [4,7,11]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [2, 8, 12, 15]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [2, 8, 12, 15]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [5, 16, 17]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [5, 16, 17]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [3, 9, 13, 18]),
                                                           create_prob_19_constraint_list(num_ele = 18, indices_list = [3, 9, 13, 18])]),
                      'constraint_rhs_array' : np.array([[6.], [6.], [6.],
                                                         [6.], [6.],
                                                         [4.], [8.], [4.], [4.],
                                                         [5.], [5.], [5.], [5.],
                                                         [3.], [3.], [8.],
                                                         [6.], [2.],
                                                         [8.],
                                                         [8.],
                                                         [8.],
                                                         [8.],
                                                         [7.],
                                                         [7.],
                                                         [14.], [14.],
                                                         [14.], [14.],
                                                         [14.], [14.],
                                                         [14.], [14.],
                                                         [14.], [14.],]),
                      'minimization_flag' : False,
                      'is_std_flag' : False,
                      'slack_var_sign_array' : create_prob_19_slack_var_array(num_constraints = 34, negative_indices = [18, 19, 20, 21, 22, 23, 25, 27, 29, 31, 33])
                      }


Problem_19_bm_dict = {'num_opt_vars' : 18,
                   'cost_coeff_array' : np.array([[6., 6., 6.,
                                                  6.1, 6.1,
                                                  5.9, 5.9 , 5.9 , 5.9,
                                                  5.8 , 5.8 , 5.8 , 5.8,
                                                  6.8 , 6.8 , 6.8,
                                                  7.3 , 7.3]]),
                   'constraint_coeff_array' : np.array([create_prob_19_constraint_list(num_ele = 18, indices_list = [1]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [2]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [3]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [4]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [5]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [6]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [7]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [8]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [9]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [10]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [11]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [12]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [13]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [14]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [15]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [16]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [17]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [18]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [1, 2, 3], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [4, 5], sign = -1.), ##toinvestigate
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [6, 7, 8, 9], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [10, 11, 12, 13], sign = -1.), ##toinvestigate
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [14, 15, 16], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [17, 18], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [1,6,10,14]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [1,6,10,14], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [4,7,11]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [4,7,11], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [2, 8, 12, 15]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [2, 8, 12, 15], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [5, 16, 17]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [5, 16, 17], sign = -1.),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [3, 9, 13, 18]),
                                                        create_prob_19_constraint_list(num_ele = 18, indices_list = [3, 9, 13, 18], sign = -1.)]),
                   'constraint_rhs_array' : np.array([[6.], [6.], [6.],
                                                      [6.], [6.],
                                                      [4.], [8.], [4.], [4.],
                                                      [5.], [5.], [5.], [5.],
                                                      [3.], [3.], [8.],
                                                      [6.], [2.],
                                                      [-8.],
                                                      [-8.],
                                                      [-8.],
                                                      [-8.],
                                                      [-7.],
                                                      [-7.],
                                                      [14.], [-14.],
                                                      [14.], [-14.],
                                                      [14.], [-14.],
                                                      [14.], [-14.],
                                                      [14.], [-14.],]),
                   'minimization_flag' : False,
                   'is_std_flag' : False,
                   'slack_var_sign_array' : create_prob_19_slack_var_array(num_constraints = 34, negative_indices = [18, 19, 20, 21, 22, 23, 25, 27, 29, 31, 33])
                   }


BJS_3_6_simplex_problem_dict = {'num_opt_vars' : 4,
                                'cost_coeff_array' : np.array([[-3., 1., 0., 0.]]),
                                'constraint_coeff_array' : np.array([[1., 2., 1., 0.],
                                                                     [1., 2., 1., 0.],
                                                                     [-1, 1., 0., 1.],
                                                                     [-1, 1., 0., 1.]]),
                                'constraint_rhs_array' : np.array([[4.], [4.], [1.] ,[1.]]),
                                'minimization_flag' : True,
                                'is_std_flag' : False,
                                'slack_var_sign_array' : np.array([[1.], [-1.], [1.], [-1.]])
                                }

BJS_3_6_bm_problem_dict = {'num_opt_vars' : 4,
                                'cost_coeff_array' : np.array([[-3., 1., 0., 0.]]),
                                'constraint_coeff_array' : np.array([[1., 2., 1., 0.],
                                                                     [-1., -2., -1., 0.],
                                                                     [-1, 1., 0., 1.],
                                                                     [1, -1., 0., -1.]]),
                                'constraint_rhs_array' : np.array([[4.], [-4.], [1.] ,[-1.]]),
                                'minimization_flag' : True,
                                'is_std_flag' : False,
                                'slack_var_sign_array' : np.array([[1.], [-1.], [1.], [-1.]])
                                }

BJS_3_7_simplex_bm_problem_dict = {'num_opt_vars' : 2,
                                'cost_coeff_array' : np.array([[ -1., -3.]]),
                                'constraint_coeff_array' : np.array([[1., -2.],
                                                                     [-1., 1.]]),
                                'constraint_rhs_array' : np.array([[4.], [3.]]),
                                'minimization_flag' : True,
                                'is_std_flag' : False,
                                'slack_var_sign_array' : np.array([[1.], [1.]])
                                }

BJS_4_4_simplex_problem_dict = {'num_opt_vars' : 2,
                                   'cost_coeff_array' : np.array([[ -3., 4.]]),
                                   'constraint_coeff_array' : np.array([[1., 1.],
                                                                        [2., 3.]]),
                                   'constraint_rhs_array' : np.array([[4.], [18.]]),
                                   'minimization_flag' : True,
                                   'is_std_flag' : False,
                                   'slack_var_sign_array' : np.array([[1.], [-1.]])
                                   }

BJS_4_4_bm_problem_dict = {'num_opt_vars' : 2,
                                'cost_coeff_array' : np.array([[ -3., 4.]]),
                                'constraint_coeff_array' : np.array([[1., 1.],
                                                                     [-2., -3.]]),
                                'constraint_rhs_array' : np.array([[4.], [-18.]]),
                                'minimization_flag' : True,
                                'is_std_flag' : False,
                                'slack_var_sign_array' : np.array([[1.], [-1.]])
                                }

BJS_4_5_simplex_problem_dict = {'num_opt_vars' : 3,
                                'cost_coeff_array' : np.array([[ -1., 2., -3.]]),
                                'constraint_coeff_array' : np.array([[1., 1., 1.],
                                                                     [-1., 1., 2.],
                                                                     [0.,2., 3.],
                                                                     [0.,0.,1.]]),
                                'constraint_rhs_array' : np.array([[6.],
                                                                   [4.],
                                                                   [10.],
                                                                   [2.]]),
                                'minimization_flag' : True,
                                'is_std_flag' : False,
                                'slack_var_sign_array' : np.array([[0.],
                                                                   [0.],
                                                                   [0.],
                                                                   [1.]])
                                }

BJS_4_5_bm_problem_dict = {'num_opt_vars' : 3,
                                'cost_coeff_array' : np.array([[ -1., 2., -3.]]),
                                'constraint_coeff_array' : np.array([[1., 1., 1.],
                                                                     [-1., -1., -1.],
                                                                     [-1., 1., 2.],
                                                                     [1., -1., -2.],
                                                                     [0.,2., 3.],
                                                                     [0.,-2., -3.],
                                                                     [0.,0.,1.]]),
                                'constraint_rhs_array' : np.array([[6.], [-6.],
                                                                   [4.], [-4.],
                                                                   [10.], [-10.],
                                                                   [2.]]),
                                'minimization_flag' : True,
                                'is_std_flag' : False,
                                'slack_var_sign_array' : np.array([[1.], [-1.],
                                                                   [1.], [-1.],
                                                                   [1.], [-1.],
                                                                   [1.]])
                                }


BJS_4_11_simplex_problem_dict = {'num_opt_vars' : 7,
                                 'cost_coeff_array' : np.array([[ 0., 0., 0., -3./4., 20., -1./2., 6.]]),
                                 'constraint_coeff_array' : np.array([[1., 0., 0., 1./4., -8., -1., 9.],
                                                                      [0., 1., 0., 1./2., -12., -1./2., 3.],
                                                                      [0., 0., 1., 0., 0., 1., 0.]]),
                                 'constraint_rhs_array' : np.array([[0.], [0.], [1.]]),
                                 'minimization_flag' : True,
                                 'is_std_flag' : False,
                                 'slack_var_sign_array' : np.array([[0.], [0.], [0.]])
                                 }

BJS_4_11_bm_problem_dict = {'num_opt_vars' : 7,
                                 'cost_coeff_array' : np.array([[ 0., 0., 0., -3./4., 20., -1./2., 6.]]),
                                 'constraint_coeff_array' : np.array([[1., 0., 0., 1./4., -8., -1., 9.],
                                                                      [-1., 0., 0., -1./4., 8., 1., -9.],
                                                                      [0., 1., 0., 1./2., -12., -1./2., 3.],
                                                                      [0., -1., 0., -1./2., 12., 1./2., -3.],
                                                                      [0., 0., 1., 0., 0., 1., 0.],
                                                                      [0., 0., -1., 0., 0., -1., 0.]]),
                                 'constraint_rhs_array' : np.array([[0.], [0.], [0.], [0.], [1.], [-1.]]),
                                 'minimization_flag' : True,
                                 'is_std_flag' : False,
                                 'slack_var_sign_array' : np.array([[0.], [0.], [0.]])
                                 }