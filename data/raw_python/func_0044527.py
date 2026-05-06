def __exportUsers(self, sort, limit=0):
        """Export the users to a dictionary.

        :param sort: field to sort the users
        :type sort: str.
        :return: exported users.
        :rtype: dict.
        """
        position = 1
        dataUsers = self.getSortedUsers(sort)

        if limit:
            dataUsers = dataUsers[:limit]

        exportedUsers = []

        for u in dataUsers:
            userExported = u.export()
            userExported["position"] = position
            exportedUsers.append(userExported)

            if position < len(dataUsers):
                userExported["comma"] = True

            position += 1
        return exportedUsers