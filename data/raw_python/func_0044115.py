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
        if order == "contributions":
            self.__users.sort(key=lambda u: u["contributions"],
                              reverse=True)
        elif order == "public":
            self.__users.sort(key=lambda u: u["public"],
                              reverse=True)
        elif order == "private":
            self.__users.sort(key=lambda u: u["private"],
                              reverse=True)
        elif order == "name":
            self.__users.sort(key=lambda u: u["name"], reverse=True)
        elif order == "followers":
            self.__users.sort(key=lambda u: u["followers"], reverse=True)
        elif order == "join":
            self.__users.sort(key=lambda u: u["join"], reverse=True)
        elif order == "organizations":
            self.__users.sort(key=lambda u: u["organizations"],
                              reverse=True)
        elif order == "repositories":
            self.__users.sort(key=lambda u: u["repositories"],
                              reverse=True)
        return self.__users