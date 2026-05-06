def factory1D(dependent_variables,
                  helper_functions):
        """Fields factory generating specialized container build around a
          triflow Model and xarray.
          Wrapper for 1D data.

          Parameters
          ----------
          dependent_variables : iterable for str
              name of the dependent variables
          helper_functions : iterable of str
              name of the helpers functions

          Returns
          -------
          triflow.BaseFields
              Specialized container which expose the data as a structured
              numpy array
          """
        return BaseFields.factory(("x", ),
                                  [(name, ("x", ))
                                   for name
                                   in dependent_variables],
                                  [(name, ("x", ))
                                   for name
                                   in helper_functions],)