def authenticate(self, username, password, is_ldap_user):
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
        user_map['is_ldap_user'] = is_ldap_user

        code, xml = self.submit({'user': user_map}, 'POST', 'authenticate/')

        return self.response(code, xml)