def inserir(self, id_user, id_group):
        """Create a relationship between User and Group.

        :param id_user: Identifier of the User. Integer value and greater than zero.
        :param id_group: Identifier of the Group. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::
            {'user_group': {'id': < id_user_group >}}

        :raise InvalidParameterError: The identifier of User or Group is null and invalid.
        :raise GrupoUsuarioNaoExisteError: UserGroup not registered.
        :raise UsuarioNaoExisteError: User not registered.
        :raise UsuarioGrupoError: User already registered in the group.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_user):
            raise InvalidParameterError(
                u'The identifier of User is invalid or was not informed.')

        if not is_valid_int_param(id_group):
            raise InvalidParameterError(
                u'The identifier of Group is invalid or was not informed.')

        url = 'usergroup/user/' + \
            str(id_user) + '/ugroup/' + str(id_group) + '/associate/'

        code, xml = self.submit(None, 'PUT', url)