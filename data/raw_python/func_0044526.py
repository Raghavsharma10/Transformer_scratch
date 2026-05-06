def getSortedUsers(self, order="public"):
        """Return a list with sorted users.

        :param order: the field to sort the users.
            - contributions (total number of contributions)
            - public (public contributions)
            - private (private contributions)
            - name
            - followers
            - join
            - organizations
            - repositories
        :type order: str.
        :return: a list of the github users sorted by the selected field.
        :rtype: str.
        """
        try:
            self.__processedUsers.sort(key=lambda u: getattr(u, order), reverse=True)
        except AttributeError:
            pass
        return self.__processedUsers