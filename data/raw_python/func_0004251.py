def remover(self, id_user_group):
        """Removes a user group by its id.

        :param id_user_group: User Group's identifier. Valid integer greater than zero.

        :return: None

        :raise GrupoUsuarioNaoExisteError: User Group not found.
        :raise InvalidParameterError: User Group id is invalid or none.
        :raise DataBaseError: Networkapi failed to access database.
        :raise XMLError: Networkapi fails generating response XML.
        """
        if not is_valid_int_param(id_user_group):
            raise InvalidParameterError(
                u'Invalid or inexistent user group id.')

        url = 'ugroup/' + str(id_user_group) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)