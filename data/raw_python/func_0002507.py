def get_keys(self, models, key=None):
        """
        Get all the primary keys for an array of models.

        :type models: list
        :type key: str

        :rtype: list
        """
        return list(set(map(lambda value: value.get_attribute(key) if key else value.get_key(), models)))