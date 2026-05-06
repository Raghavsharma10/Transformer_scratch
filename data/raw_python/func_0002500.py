def add_eager_constraints(self, models):
        """
        Set the constraints for an eager load of the relation.

        :type models: list
        """
        super(MorphOneOrMany, self).add_eager_constraints(models)

        self._query.where(self._morph_type, self._morph_class)