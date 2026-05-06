def find(self, objects):
        """Find exactly one match in the list of objects.

        :param objects: objects to filter
        :type objects: :class:`list`
        :return: the one matching object
        :raises groupy.exceptions.NoMatchesError: if no objects match
        :raises groupy.exceptions.MultipleMatchesError: if multiple objects match
        """
        matches = list(self.__call__(objects))
        if not matches:
            raise exceptions.NoMatchesError(objects, self.tests)
        elif len(matches) > 1:
            raise exceptions.MultipleMatchesError(objects, self.tests,
                                                  matches=matches)
        return matches[0]