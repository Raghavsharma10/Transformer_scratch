def match(self, models, results, relation):
        """
        Match the eagerly loaded results to their parents.

        :type models: list
        :type results: Collection
        :type relation:  str
        """
        foreign = self._foreign_key

        other = self._other_key

        dictionary = {}

        for result in results:
            dictionary[result.get_attribute(other)] = result

        for model in models:
            value = model.get_attribute(foreign)

            if value in dictionary:
                model.set_relation(relation, dictionary[value])

        return models