def _new_pivot_query(self):
        """
        Create a new query builder for the pivot table.

        :rtype: eloquent.orm.Builder
        """
        query = super(MorphToMany, self)._new_pivot_query()

        return query.where(self._morph_type, self._morph_class)