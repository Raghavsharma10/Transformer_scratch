def _parse_relations(self, relations):
        """
        Parse a list of relations into individuals.

        :param relations: The relation to parse
        :type relations: list

        :rtype: dict
        """
        results = {}

        for relation in relations:
            if isinstance(relation, dict):
                name = list(relation.keys())[0]
                constraints = relation[name]
            else:
                name = relation
                constraints = self.__class__(self.get_query().new_query())

            results = self._parse_nested(name, results)

            results[name] = constraints

        return results