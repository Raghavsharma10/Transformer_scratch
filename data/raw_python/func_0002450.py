def _match_one_or_many(self, models, results, relation, type):
        """
        Match the eargerly loaded resuls to their single parents.

        :param models: The parents
        :type models: list

        :param results: The results collection
        :type results: Collection

        :param relation: The relation
        :type relation: str

        :param type: The match type
        :type type: str

        :rtype: list
        """
        dictionary = self._build_dictionary(results)

        for model in models:
            key = model.get_attribute(self._local_key)

            if key in dictionary:
                value = self._get_relation_value(dictionary, key, type)

                model.set_relation(relation, value)

        return models