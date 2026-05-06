def search(self, id_perm):
        """Search Administrative Permission from by the identifier.

        :param id_perm: Identifier of the Administrative Permission. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {'perm': {'ugrupo': < ugrupo_id >,
            'permission': < permission_id >, 'id': < id >,
            'escrita': < escrita >, 'leitura': < leitura >}}

        :raise InvalidParameterError: Group User identifier is null and invalid.
        :raise PermissaoAdministrativaNaoExisteError: Administrative Permission not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_perm):
            raise InvalidParameterError(
                u'The identifier of Administrative Permission is invalid or was not informed.')

        url = 'aperms/get/' + str(id_perm) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)