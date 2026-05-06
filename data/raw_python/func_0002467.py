def eager_load_relations(self, models):
        """
        Eager load the relationship of the models.

        :param models:
        :type models: list

        :return: The models
        :rtype: list
        """
        for name, constraints in self._eager_load.items():
            if name.find('.') == -1:
                models = self._load_relation(models, name, constraints)

        return models