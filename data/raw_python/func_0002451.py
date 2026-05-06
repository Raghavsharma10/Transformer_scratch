def _get_relation_value(self, dictionary, key, type):
        """
        Get the value of the relationship by one or many type.

        :type dictionary: dict
        :type key: str
        :type type: str
        """
        value = dictionary[key]

        if type == 'one':
            return value[0]

        return self._related.new_collection(value)