def list_by_group(self, id_ugroup):
        """Search Administrative Permission by Group User by identifier.

        :param id_ugroup: Identifier of the Group User. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'perms': [{'ugrupo': < ugrupo_id >, 'permission':  { 'function' < function >, 'id': < id > },
            'id': < id >, 'escrita': < escrita >,
            'leitura': < leitura >}, ... ] }

        :raise InvalidParameterError: Group User is null and invalid.
        :raise UGrupoNotFoundError: Group User not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if id_ugroup is None:
            raise InvalidParameterError(
                u'The identifier of Group User is invalid or was not informed.')

        url = 'aperms/group/' + str(id_ugroup) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)