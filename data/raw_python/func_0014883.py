def get(self, names_to_get, convert_to_numpy=True):
    """ Loads the requested variables from the matlab com client.

    names_to_get can be either a variable name or a list of variable names.
    If it is a variable name, the values is returned.
    If it is a list, a dictionary of variable_name -> value is returned.

    If convert_to_numpy is true, the method will all array values to numpy
    arrays. Scalars are left as regular python objects.

    """
    self._check_open()
    single_itme = isinstance(names_to_get, (unicode, str))
    if single_itme:
      names_to_get = [names_to_get]
    ret = {}
    for name in names_to_get:
      ret[name] = self.client.GetWorkspaceData(name, 'base')
      # TODO(daniv): Do we really want to reduce dimensions like that? what if this a row vector?
      while isinstance(ret[name], (tuple, list)) and len(ret[name]) == 1:
        ret[name] = ret[name][0]
      if convert_to_numpy and isinstance(ret[name], (tuple, list)):
        ret[name] = np.array(ret[name])
    if single_itme:
      return ret.values()[0]
    return ret