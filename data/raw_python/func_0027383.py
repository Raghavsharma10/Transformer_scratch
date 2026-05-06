def __set_basic_auth_string(self, username, password):
        '''
        Creates and sets the authentication string for (write-)accessing the
            Handle Server. No return, the string is set as an attribute to
            the client instance.

        :param username: Username handle with index: index:prefix/suffix.
        :param password: The password contained in the index of the username
            handle.
        '''
        auth = b2handle.utilhandle.create_authentication_string(username, password)
        self.__basic_authentication_string = auth