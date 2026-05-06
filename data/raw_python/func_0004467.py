def remover(self, id_perms):
        """Remove Administrative Permission from by the identifier.

        :param id_perms: Identifier of the Administrative Permission. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Administrative Permission is null and invalid.
        :raise PermissaoAdministrativaNaoExisteError: Administrative Permission not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_perms):
            raise InvalidParameterError(
                u'The identifier of Administrative Permission is invalid or was not informed.')

        url = 'aperms/' + str(id_perms) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)