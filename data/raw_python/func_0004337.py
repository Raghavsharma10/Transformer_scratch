def change_password(self, id_user, user_current_password, password):
        """Change password of User from by the identifier.

        :param id_user: Identifier of the User. Integer value and greater than zero.
        :param user_current_password: Senha atual do usuário.
        :param password: Nova Senha do usuário.

        :return: None

        :raise UsuarioNaoExisteError: User not registered.
        :raise InvalidParameterError: The identifier of User is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.

        """
        if not is_valid_int_param(id_user):
            raise InvalidParameterError(
                u'The identifier of User is invalid or was not informed.')

        if password is None or password == "":
            raise InvalidParameterError(
                u'A nova senha do usuário é inválida ou não foi informada')

        user_map = dict()
        user_map['user_id'] = id_user
        user_map['password'] = password

        code, xml = self.submit(
            {'user': user_map}, 'POST', 'user-change-pass/')

        return self.response(code, xml)