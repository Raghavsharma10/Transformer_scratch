def _generate_sympify_namespace(
    independent_variables, dependent_variables, helper_functions
):
    """Generate the link between the symbols of the derivatives and the
      sympy Derivative operation.

      Parameters
      ----------
      independent_variable : str
          name of the independant variable ("x")
      dependent_variables : iterable of str
          names of the dependent variables
      helper_functions : iterable of str
          names of the helper functions

      Returns
      -------
      dict
          dictionnary containing the symbol to parse as keys and the sympy expression to evaluate instead as values.
      """  # noqa

    independent_variable = independent_variables[0]  # TEMP FIX BEFORE REAL ND
    symbolic_independent_variable = Symbol(independent_variable)

    def partial_derivative(symbolic_independent_variable, i, expr):
        return Derivative(expr, symbolic_independent_variable, i)

    namespace = {independent_variable: symbolic_independent_variable}
    namespace.update(
        {
            "d%s"
            % (independent_variable * i): partial(
                partial_derivative, symbolic_independent_variable, i
            )
            for i in range(1, 10)
        }
    )
    namespace.update(
        {
            "d%s%s"
            % (independent_variable * order, var): Derivative(
                Function(var)(independent_variable), independent_variable, order
            )
            for order, var in product(
                range(1, 10), dependent_variables + helper_functions
            )
        }
    )
    logging.debug("sympy namespace: %s" % namespace)
    return namespace