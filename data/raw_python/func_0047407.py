def _extract_params_from_i0(only_variable_parameters, parameters_with_variability, initial_conditions_with_variability):
    """
    Used within the distance/cost function to create the current kinetic parameter and initial condition vectors
    to be used during that interaction, using current values in i0.

    This function takes i0 and complements it with additional information from variables that we do not want to vary
    so the simulation function could be run and values compared.

    :param only_variable_parameters: `i0` list returned from `make_i0`
    :param param: list of starting values for kinetic parameters
    :param vary: list to identify which values in `param` to vary during inference (0=fixed, 1=optimise)
    :param initcond: list of starting values (i.e. at t0) for moments
    :param varyic: list to identify which values in `initcond` to vary (0=fixed, 1=optimise)
    :return:
    """

    complete_params = []
    counter = 0
    for param, is_variable in parameters_with_variability:
        # If param not variable, add it from param list
        if not is_variable:
            complete_params.append(param)
        else:
            # Otherwise add it from variable parameters list
            complete_params.append(only_variable_parameters[counter])
            counter += 1

    complete_initial_conditions = []
    for initial_condition, is_variable in initial_conditions_with_variability:
        if not is_variable:
            complete_initial_conditions.append(initial_condition)
        else:
            complete_initial_conditions.append(only_variable_parameters[counter])
            counter += 1

    return complete_params, complete_initial_conditions