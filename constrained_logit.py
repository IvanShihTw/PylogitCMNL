# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 07:19:49 2016

@name:      Constrained MultiNomial Logit
@author:    Timothy Brathwaite (Modified by Ivan Shih with CMNL extension)
@summary:   Contains functions necessary for estimating multinomial logit
            models (with the help of the "base_multinomial_cm.py" file).
            Differs from version one since it works with the shape, intercept,
            index coefficient partitioning of estimated parameters as opposed
            to the shape, index coefficient partitioning scheme of version 1.
"""
from __future__ import absolute_import

import warnings
import numpy as np
from scipy.sparse import diags

from . import cmnl_choice_calcs as cmcc
from . import base_multinomial_cm_v2 as base_mcm
from .estimation import EstimationObj
from .estimation import cmnl_estimate
from .display_names import model_type_to_display_name

# Create a variable that will be printed if there is a non-fatal error
# in the MNL class construction
_msg_1 = "The Constrained Multinomial Logit Model has no shape parameters. "
_msg_2 = "shape_names and shape_ref_pos will be ignored if passed."
_shape_ignore_msg = _msg_1 + _msg_2

# Create a warning string that will be issued if ridge regression is performed.
_msg_3 = "NOTE: An L2-penalized regression is being performed. The "
_msg_4 = "reported standard errors and robust standard errors "
_msg_5 = "***WILL BE INCORRECT***."
_ridge_warning_msg = _msg_3 + _msg_4 + _msg_5

# Alias necessary functions from the base multinomial choice model module
general_log_likelihood = cmcc.calc_log_likelihood
general_gradient = cmcc.calc_gradient
general_calc_probabilities = cmcc.calc_probabilities
general_hessian = cmcc.calc_hessian

#############################CMNL split phi_scale, upper_limit per constrained var######################
def split_param_vec(beta,
                    phi_scale,
                    rows_to_alts=None,
                    design=None,
                    return_all_types=False,
                    *args, **kwargs):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features).
    phi_scale :
    upper_limit :  
    rows_to_alts : None,
        Not actually used. Included merely for consistency with other models.
    design : None.
        Not actually used. Included merely for consistency with other models.
    return_all_types : bool, optional.
        Determines whether or not a tuple of 4 elements will be returned (with
        one element for the nest, shape, intercept, and index parameters for
        this model). If False, a tuple of 3 elements will be returned, as
        described below.

    Returns
    -------
    tuple.
        `(None, None, beta)`. This function is merely for compatibility with
        the other choice model files.

    Note
    ----
    If `return_all_types == True` then the function will return a tuple of four
    objects. In order, these objects will either be None or the arrays
    representing the arrays corresponding to the nest, shape, intercept, and
    index parameters.
    """
    if return_all_types:
        return None, None, None, beta, phi_scale
    else:
        return None, None, beta, phi_scale


def _cmnl_utility_transform(systematic_utilities, *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        Should contain the systematic utilities for each each available
        alternative for each observation.

    Returns
    -------
    `systematic_utilities[:, None]`
    """
    # Be sure to return a 2D array since other functions will be expecting this
    if len(systematic_utilities.shape) == 1:
        systematic_utilities = systematic_utilities[:, np.newaxis]

    return systematic_utilities


def _cmnl_transform_deriv_c(*args, **kwargs):
    """
    Returns None.

    This is a place holder function since the MNL model has no shape
    parameters.
    """
    # This is a place holder function since the MNL model has no shape
    # parameters.
    return None


def _cmnl_transform_deriv_alpha(*args, **kwargs):
    """
    Returns None.

    This is a place holder function since the MNL model has no intercept
    parameters outside of the index.
    """
    # This is a place holder function since the MNL model has no intercept
    # parameters outside the index.
    return None

class CMNLEstimator(EstimationObj):
    """
    Estimation Object used to enforce uniformity in the estimation process
    across the various logit-type models.

    Parameters
    ----------
    model_obj : a pylogit.base_multinomial_cm_v2.MNDC_Model instance.
        Should contain the following attributes:

          - alt_IDs
          - choices
          - design
          - intercept_ref_position
          - shape_ref_position
          - utility_transform
    mapping_res : dict.
        Should contain the scipy sparse matrices that map the rows of the long
        format dataframe to various other objects such as the available
        alternatives, the unique observations, etc. The keys that it must have
        are `['rows_to_obs', 'rows_to_alts', 'chosen_row_to_obs']`
    ridge : int, float, long, or None.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. The scalar should be greater than or equal to
            zero..
    zero_vector : 1D ndarray.
        Determines what is viewed as a "null" set of parameters. It is
        explicitly passed because some parameters (e.g. parameters that must be
        greater than zero) have their null values at values other than zero.
    split_params : callable.
        Should take a vector of parameters, `mapping_res['rows_to_alts']`, and
        model_obj.design as arguments. Should return a tuple containing
        separate arrays for the model's shape, outside intercept, and index
        coefficients. For each of these arrays, if this model does not contain
        the particular type of parameter, the callable should place a `None` in
        its place in the tuple.
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are
        not to change from their initial values. If a list is passed, the
        elements are to be integers where no such integer is greater than
        `num_params` Default == None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.
    """    
    def set_derivatives(self):     ##########################################################
        # Pre-calculate the derivative of the transformation vector with
        # respect to the vector of systematic utilities #####################(derivatives of beta)
        dh_dv = diags(np.ones(self.design.shape[0]), 0, format='csr')

        # Create a function to calculate dh_dv which will return the
        # pre-calculated result when called
        def calc_dh_dv(*args):
            return dh_dv

        self.calc_dh_dv = calc_dh_dv
        self.calc_dh_d_alpha = _cmnl_transform_deriv_alpha
        self.calc_dh_d_shape = _cmnl_transform_deriv_c
                
        ##############################################
    def check_length_of_initial_values(self, init_values):
        """
        Ensures that `init_values` is of the correct length. Raises a helpful
        ValueError if otherwise.

        Parameters
        ----------
        init_values : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each index coefficient, outside intercept
            parameter, and shape parameter being estimated.

        Returns
        -------
        None.
        """
        # Calculate the expected number of index parameters
        num_index_coefs = self.design.shape[1]

        if init_values.shape[0] != num_index_coefs:
            msg_1 = "The initial values are of the wrong dimension."
            msg_2 = "It should be of dimension {}"
            msg_3 = "But instead it has dimension {}"
            raise ValueError(msg_1 +
                             msg_2.format(num_index_coefs) +
                             msg_3.format(init_values.shape[0]))

        return None

        ##############################################
    def check_length_of_cmnl_initial_values(self, phi_scale_init_vals):
        """
        Ensures that `init_values` is of the correct length. Raises a helpful
        ValueError if otherwise.

        Parameters
        ----------
        init_values : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each index coefficient, outside intercept
            parameter, and shape parameter being estimated.

        Returns
        -------
        None.
        """
        # Calculate the expected number of index parameters
        num_cmnl_vars = self.constraining_vars.shape[1]

        if phi_scale_init_vals.shape[0] != num_cmnl_vars:
            msg_1 = "The initial values are of the wrong dimension."
            msg_2 = "It should be of dimension {}"
            msg_3 = "But instead it has dimension {}"
            raise ValueError(msg_1 +
                             msg_2.format(num_cmnl_vars) +
                             msg_3.format(phi_scale_init_vals.shape[0]))

        return None    

    def cmnl_convenience_calc_probs(self, betas, phi_scale):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        # shapes, intercepts, betas, phi_scale, upper_limit = self.cmnl_convenience_split_params(betas, phi_scale, upper_limit)
        shapes = None
        intercepts = None

        prob_args = [betas,
                     self.design,
                     self.constraining_vars_vals,
                     self.upper_limit_input,
                     phi_scale,
                     self.cutoff_tol,
                     self.alt_id_vector,
                     self.rows_to_obs,
                     self.rows_to_alts,
                     self.utility_transform]

        prob_kwargs = {"intercept_params": intercepts,
                       "shape_params": shapes,
                       "chosen_row_to_obs": self.chosen_row_to_obs,
                       "return_long_probs": True}
        prob_results = cmcc.calc_probabilities(*prob_args, **prob_kwargs)

        return prob_results

    def cmnl_convenience_calc_log_likelihood(self, betas, phi_scale):
        """
        Calculates the log-likelihood for this model and dataset.
        """
        # shapes, intercepts, betas, phi_scale, upper_limit = self.cmnl_convenience_split_params(betas, phi_scale, upper_limit)
        shapes = None
        intercepts = None

        args = [betas,
                self.design,
                self.constraining_vars_vals,
                self.upper_limit_input,
                phi_scale,
                self.cutoff_tol,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.choice_vector,
                self.utility_transform]

        kwargs = {"intercept_params": intercepts,
                  "shape_params": shapes,
                  "ridge": self.ridge,
                  "weights": self.weights}
        log_likelihood = cmcc.calc_log_likelihood(*args, **kwargs)

        return log_likelihood

    def cmnl_convenience_calc_gradient(self, betas, phi_scale):
        """
        Calculates the gradient of the log-likelihood for this model / dataset.
        """
        # shapes, intercepts, betas, phi_scale, upper_limit = self.cmnl_convenience_split_params(betas, phi_scale, upper_limit)
        shapes = None
        intercepts = None

        args = [betas,
                self.design,
                self.constraining_vars_vals,
                self.upper_limit_input,
                phi_scale,
                self.cutoff_tol,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.choice_vector,
                self.utility_transform,
                self.calc_dh_d_shape,
                self.calc_dh_dv,
                self.calc_dh_d_alpha,
                intercepts,
                self.ridge,
                self.weights,
                shapes]

        return cmcc.calc_gradient(*args)

    def cmnl_convenience_calc_hessian(self, betas, phi_scale):
        """
        Calculates the hessian of the log-likelihood for this model / dataset.
        """
        # shapes, intercepts, betas, phi_scale, upper_limit = self.cmnl_convenience_split_params(betas, phi_scale, upper_limit)
        shapes = None
        intercepts = None

        args = [betas,
                self.design,
                self.constraining_vars_vals,
                self.upper_limit_input,
                phi_scale,
                self.cutoff_tol,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.utility_transform,
                self.calc_dh_d_shape,
                self.calc_dh_dv,
                self.calc_dh_d_alpha,
                intercepts,
                shapes,
                self.ridge,
                self.weights]

        return cmcc.calc_hessian(*args)

    def cmnl_convenience_calc_fisher_approx(self, betas, phi_scale):
        """
        Calculates the BHHH approximation of the Fisher Information Matrix for
        this model / dataset.
        """
        # shapes, intercepts, betas, phi_scale, upper_limit = self.cmnl_convenience_split_params(betas, phi_scale, upper_limit)
        shapes = None
        intercepts = None

        args = [betas,
                self.design,
                self.constraining_vars_vals,
                self.upper_limit_input,
                phi_scale,
                self.cutoff_tol,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.choice_vector,
                self.utility_transform,
                self.calc_dh_d_shape,
                self.calc_dh_dv,
                self.calc_dh_d_alpha,
                intercepts,
                shapes,
                self.ridge,
                self.weights]

        return cmcc.calc_fisher_info_matrix(*args)    


class CMNL(base_mcm.MNDC_Model):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col :str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:
            - if the corresponding value in `specification` is "all_same", then
              there should be a single string as the value in names.
            - if the corresponding value in `specification` is "all_diff", then
              there should be a list of strings as the value in names. There
              should be one string in the value in names for each possible
              alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.
        Default == None.

    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 names=None,
                 cmnl_spec=None,
                 cutoff_tol=None,
                 upper_limit_input=None,
                 num_index=None,
                 num_constrained=None,
                 *args, **kwargs):
        ##########
        # Print a helpful message for users who have included shape parameters
        # or shape names unneccessarily
        ##########
        for keyword in ["shape_names", "shape_ref_pos"]:
            if keyword in kwargs and kwargs[keyword] is not None:
                warnings.warn(_shape_ignore_msg)
                break

        if "intercept_ref_pos" in kwargs:
            if kwargs["intercept_ref_pos"] is not None:
                msg = "The CMNL model should have all intercepts in the index."
                raise ValueError(msg)

        # Carry out the common instantiation process for all choice models
        super(CMNL, self).__init__(data,
                                  alt_id_col,
                                  obs_id_col,
                                  choice_col,
                                  specification,
                                  names=names,
                                  model_type=model_type_to_display_name["CMNL"],
                                  cmnl_spec=cmnl_spec,
                                  cutoff_tol=cutoff_tol,
                                  upper_limit_input=upper_limit_input,
                                  num_index=num_index,
                                  num_constrained=num_constrained)

        # Store the utility transform function
        self.utility_transform = _cmnl_utility_transform

        return None

    def fit_mle(self,
                init_vals,
                phi_scale_init_vals,
                print_res=True,
                method="trust-constr",
                loss_tol=1e-06,
                gradient_tol=1e-06,
                maxiter=2000,
                ridge=None,
                constrained_pos=None,
                just_point=False,
                **kwargs):
        """
        Parameters
        ----------
        init_vals : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each utility coefficient being estimated.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string that can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm that
            is used for this problem. If 'em' is passed, a custom coded EM
            algorithm will be used. Default `== 'newton-cg'`.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next that is needed to determine
            convergence. Default `== 1e-06`.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. Default `== None`.
        constrained_pos : list or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_vals.size.` Default == None.
        just_point : bool, optional.
            Determines whether (True) or not (False) calculations that are non-
            critical for obtaining the maximum likelihood point estimate will
            be performed. If True, this function will return the results
            dictionary from scipy.optimize. Default == False.

        Returns
        -------
        None or dict.
            If `just_point` is False, None is returned and the estimation
            results are saved to the model instance. If `just_point` is True,
            then the results dictionary from scipy.optimize() is returned.
        """
        # Check integrity of passed arguments
        kwargs_to_be_ignored = ["init_shapes", "init_intercepts", "init_coefs"]
        if any([x in kwargs for x in kwargs_to_be_ignored]):
            msg = "MNL model does not use of any of the following kwargs:\n{}"
            msg_2 = "Remove such kwargs and pass a single init_vals argument"
            raise ValueError(msg.format(kwargs_to_be_ignored) + msg_2)

        if ridge is not None:
            warnings.warn(_ridge_warning_msg)

        # Store the optimization method
        self.optimization_method = method

        # Store the ridge parameter
        self.ridge_param = ridge

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        mapping_res = self.get_mappings_for_fit()

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        # Create the estimation object for CMNL specific
        cmnl_ps_zero_vector = np.zeros(phi_scale_init_vals.shape) + 0.001
        estimator_args = [self,
                          mapping_res,
                          ridge,
                          zero_vector,
                          split_param_vec,
                          cmnl_ps_zero_vector]
        estimator_kwargs = {"constrained_pos": constrained_pos}
        cmnl_estimator = CMNLEstimator(*estimator_args,
                                       **estimator_kwargs)
        # Set the derivative functions for estimation
        cmnl_estimator.set_derivatives()

        ##################CMNL will have more param than MNL#############################        
        # Perform one final check on the length of the initial values
        cmnl_estimator.check_length_of_initial_values(init_vals)
        #cmnl_estimator.check_length_of_cmnl_initial_values(cmnl_init_vals)
        
        # Get the estimation results
        estimation_res = cmnl_estimate(init_vals,
                                       phi_scale_init_vals,
                                       cmnl_estimator,
                                       method,
                                       loss_tol,
                                       gradient_tol,
                                       maxiter,
                                       print_res,
                                       just_point=just_point)

        if not just_point:
            # Store the estimation results
            self.store_fit_results(estimation_res)

            return None
        else:
            return estimation_res
