def _load_relation(self, models, name, constraints):
        """
        Eagerly load the relationship on a set of models.

        :rtype: list
        """
        relation = self.get_relation(name)

        relation.add_eager_constraints(models)

        if callable(constraints):
            constraints(relation)
        else:
            relation.merge_query(constraints)

        models = relation.init_relation(models, name)

        results = relation.get_eager()

        return relation.match(models, results, name)