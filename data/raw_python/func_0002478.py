def _parse_nested(self, name, results):
        """
        Parse the nested relationship in a relation.

        :param name: The name of the relationship
        :type name: str

        :type results: dict

        :rtype: dict
        """
        progress = []

        for segment in name.split('.'):
            progress.append(segment)

            last = '.'.join(progress)
            if last not in results:
                results[last] = self.__class__(self.get_query().new_query())

        return results