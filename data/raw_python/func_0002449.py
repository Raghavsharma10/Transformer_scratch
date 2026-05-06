def match_many(self, models, results, relation):
        """
        Match the eargerly loaded resuls to their single parents.

        :param models: The parents
        :type models: list

        :param results: The results collection
        :type results: Collection

        :param relation: The relation
        :type relation: str

        :rtype: list
        """
        return self._match_one_or_many(models, results, relation, 'many')