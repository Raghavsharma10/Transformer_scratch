def alterar(self, id_user, user, password, nome, ativo, email, user_ldap):
        """Change User from by the identifier.

        :param id_user: Identifier of the User. Integer value and greater than zero.
        :param user: Username. String with a minimum 3 and maximum of 45 characters
        :param password: User password. String with a minimum 3 and maximum of 45 characters
        :param nome: User name. String with a minimum 3 and maximum of 200 characters
        :param email: User Email. String with a minimum 3 and maximum of 300 characters
        :param ativo: Status. 0 or 1
        :param user_ldap: LDAP Username. String with a minimum 3 and maximum of 45 characters

        :return: None

        :raise InvalidParameterError: The identifier of User, user, pwd, name, email or  active is null and invalid.
        :raise UserUsuarioDuplicadoError: There is already a registered user with the value of user.
        :raise UsuarioNaoExisteError: User not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.

        """
        if not is_valid_int_param(id_user):
            raise InvalidParameterError(
                u'The identifier of User is invalid or was not informed.')

        url = 'user/' + str(id_user) + '/'

        user_map = dict()
        user_map['user'] = user
        user_map['password'] = password
        user_map['name'] = nome
        user_map['active'] = ativo
        user_map['email'] = email
        user_map['user_ldap'] = user_ldap

        code, xml = self.submit({'user': user_map}, 'PUT', url)

        return self.response(code, xml)