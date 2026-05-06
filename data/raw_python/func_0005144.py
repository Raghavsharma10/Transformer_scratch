def _to_tuple(self, _list):
        """ Recursively converts lists to tuples """
        result = list()
        for l in _list:
            if isinstance(l, list):
                result.append(tuple(self._to_tuple(l)))
            else:
                result.append(l)
        return tuple(result)