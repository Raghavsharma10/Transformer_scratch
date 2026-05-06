def data(self):
        """
        return (data_dict, key) tuple instead of models instances
        """
        clone = copy.deepcopy(self)
        clone._cfg['rtype'] = ReturnType.Object
        return clone