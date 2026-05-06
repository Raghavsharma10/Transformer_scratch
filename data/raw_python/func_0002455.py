def get_relation_count_query(self, query, parent):
        """
        Add the constraints for a relationship count query.

        :type query: eloquent.orm.Builder
        :type parent: eloquent.orm.Builder

        :rtype: eloquent.orm.Builder
        """
        query = super(MorphToMany, self).get_relation_count_query(query, parent)

        return query.where('%s.%s' % (self._table, self._morph_type), self._morph_class)