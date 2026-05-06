def _generate_values_with_variability_and_constraints(self, symbols, starting_values, variable_parameters):
        """
        Generates the `values_with_variability` formatted list
        from the provided symbols, starting values and variable parameters

        :param symbols: The symbols defining each of the values in the starting values list
        :param starting_values: the actual starting values
        :param variable_parameters: a dictionary/set/list of variables that are variable
                                    if dictionary provided, the contents should be `symbol: range` where range is
                                    a tuple ``(min_val, max_val)`` of allowed parameter values or ``None`` for no limit.
                                    if set/list provided, the ranges will be assumed to be ``None`` for each of
                                    the parameters
        :type variable_parameters: dict|iterable
        :return:
        """
        values_with_variability = []
        constraints = []

        if not isinstance(variable_parameters, dict):
            # Convert non/dict representations to Dict with nones
            variable_parameters = {p: None for p in variable_parameters}

        for parameter, parameter_value in zip(symbols, starting_values):
            try:
                constraint = variable_parameters[parameter]
                variable = True
            except KeyError:
                try:
                    constraint = variable_parameters[str(parameter)]
                    variable = True
                except KeyError:
                    constraint = None
                    variable = False

            values_with_variability.append((parameter_value, variable))
            if variable:
                constraints.append(constraint)

        return values_with_variability, constraints