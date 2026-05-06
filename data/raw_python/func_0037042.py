def get_dict_value(dictionary, path):
        """ Safely get the value of a dictionary given a key path. For
            instance, for the dictionary `{ 'a': { 'b': 1 } }`, the value at
            key path ['a'] is { 'b': 1 }, at key path ['a', 'b'] is 1, at
            key path ['a', 'b', 'c'] is None.

        :param dictionary: a dictionary.
        :param path: the key path.
        :return: The value of d at the given key path, or None if the key
                 path does not exist.
        """
        if len(path) == 0:
            return None
        temp_dictionary = dictionary
        try:
            for k in path:
                temp_dictionary = temp_dictionary[k]
            return temp_dictionary
        except (KeyError, TypeError):
            pass
        return None