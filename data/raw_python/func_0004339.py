def authenticate_ldap(self, username, password):
        """Get user by username and password and their permissions.

        :param username: Username. String with a minimum 3 and maximum of 45 characters
        :param password: User password. String with a minimum 3 and maximum of 45 characters

        :return: Following dictionary:

        ::

            {'user': {'id': < id >}
            {'user': < user >}
            {'nome': < nome >}
            {'pwd': < pwd >}
            {'email': < email >}
            {'active': < active >}
            {'permission':[ {'<function>': { 'write': <value>, 'read': <value>}, ... more function ... ] } } }

        :raise InvalidParameterError: The value of username or password is invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        user_map = dict()
        user_map['username'] = username
        user_map['password'] = password

        code, xml = self.submit(
            {'user': user_map}, 'POST', 'authenticate/ldap/')

        return self.response(code, xml)