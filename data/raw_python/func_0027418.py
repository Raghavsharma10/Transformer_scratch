def __set_revlookup_auth_string(self, username, password):
        '''
        Creates and sets the authentication string for accessing the reverse
            lookup servlet. No return, the string is set as an attribute to
            the client instance.

        :param username: Username.
        :param password: Password.
        '''
        auth = b2handle.utilhandle.create_authentication_string(username, password)
        self.__revlookup_auth_string = auth