def _set_keys_for_save_query(self, query):
        """
        Set the keys for a save update query.

        :param query: A Builder instance
        :type query: eloquent.orm.Builder

        :return: The Builder instance
        :rtype: eloquent.orm.Builder
        """
        query.where(self._morph_type, self._morph_class)

        return super(MorphPivot, self)._set_keys_for_save_query(query)