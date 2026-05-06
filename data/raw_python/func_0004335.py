def inserir(self, user, pwd, name, email, user_ldap):
        """Inserts a new User and returns its identifier.

        The user will be created with active status.

        :param user: Username. String with a minimum 3 and maximum of 45 characters
        :param pwd: User password. String with a minimum 3 and maximum of 45 characters
        :param name: User name. String with a minimum 3 and maximum of 200 characters
        :param email: User Email. String with a minimum 3 and maximum of 300 characters
        :param user_ldap: LDAP Username. String with a minimum 3 and maximum of 45 characters

        :return: Dictionary with the following structure:

        ::

            {'usuario': {'id': < id_user >}}

        :raise InvalidParameterError: The identifier of User, user, pwd, name or email is null and invalid.
        :raise UserUsuarioDuplicadoError: There is already a registered user with the value of user.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        user_map = dict()
        user_map['user'] = user
        user_map['password'] = pwd
        user_map['name'] = name
        user_map['email'] = email
        user_map['user_ldap'] = user_ldap

        code, xml = self.submit({'user': user_map}, 'POST', 'user/')

        return self.response(code, xml)