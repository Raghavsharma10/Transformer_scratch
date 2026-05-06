def get_dict(self, exclude_keys=None, include_keys=None):
        """
        return dictionary of keys and values corresponding to this model's
        data - if include_keys is null the function will return all keys

        :param exclude_keys: (optional) is a list of columns from model that
        should not be returned by this function
        :param include_keys: (optional) is a list of columns from model that
        should be returned by this function
        :return:
        """
        d = {}
        exclude_keys_list = exclude_keys or []
        include_keys_list = include_keys or []
        for k in self._get_keys():
            if k not in exclude_keys_list and (
                k in include_keys_list or not include_keys
            ):
                d[k] = getattr(self, k)
        return d