def __addUser(self, new_user):
        """Add new users to the list.

        :param new_user: name of a GitHub user to include in
            the ranking
        :type new_user: str.
        """
        self.__lockReadAddUser.acquire()
        if new_user not in self.__cityUsers and \
                new_user not in self.__excludedUsers:
            self.__lockReadAddUser.release()
            self.__logger.debug("__addUser: Adding " + new_user)
            self.__cityUsers.add(new_user)

            myNewUser = GitHubUser(new_user)
            myNewUser.getData()
            myNewUser.getRealContributions()

            userLoc = myNewUser.location
            if not any(s in userLoc for s in self.__excludedLocations):
                self.__processedUsers.append(myNewUser)
        else:
            self.__logger.debug("__addUser: Excluding " + new_user)
            self.__lockReadAddUser.release()