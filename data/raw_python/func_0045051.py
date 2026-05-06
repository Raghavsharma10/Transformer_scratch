def factory(coords,
                dependent_variables,
                helper_functions):
        """Fields factory generating specialized container build around a
          triflow Model and xarray.

          Parameters
          ----------
          coords: iterable of str:
              coordinates name. First coordinate have to be shared with all
              variables
          dependent_variables : iterable tuple (name, coords)
              coordinates and name of the dependent variables
          helper_functions : iterable tuple (name, coords)
              coordinates and name of the helpers functions

          Returns
          -------
          triflow.BaseFields
              Specialized container which expose the data as a structured
              numpy array
          """
        Field = type('Field', BaseFields.__bases__,
                     dict(BaseFields.__dict__))
        Field._coords = coords
        Field.dependent_variables_info = dependent_variables
        Field.helper_functions_info = helper_functions
        Field._var_info = [*list(Field.dependent_variables_info),
                           *list(Field.helper_functions_info)]
        Field.dependent_variables = [dep[0]
                                     for dep
                                     in Field.dependent_variables_info]
        Field.helper_functions = [dep[0]
                                  for dep
                                  in Field.helper_functions_info]
        Field._keys, Field._coords_info = zip(*Field._var_info)
        return Field