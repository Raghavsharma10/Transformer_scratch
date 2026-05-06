def _build_list_params(param_name, key, values):
        """Builds a list of POST parameters from a list or single value."""
        params = {}
        if hasattr(values, '__iter__'):
            index = 0
            for value in values:
                params[str(param_name) + '[' + str(index) + '].' + str(key)] = str(value)
                index += 1
        else:
            params[str(param_name) + '[0].' + str(key)] = str(values)
        return params