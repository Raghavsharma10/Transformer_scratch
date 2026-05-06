def get_relation_count_query(self, query, parent):
        """
        Add the constraints for a relationship count query.

        :type query: Builder
        :type parent: Builder

        :rtype: Builder
        """
        query = super(MorphOneOrMany, self).get_relation_count_query(query, parent)

        return query.where(self._morph_type, self._morph_class)