def search(self, id_ugroup):
        """Search Group User by its identifier.

        :param id_ugroup: Identifier of the Group User. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {‘user_group’: {'escrita': < escrita >,
            'nome': < nome >,
            'exclusao': < exclusao >,
            'edicao': < edicao >,
            'id': < id >,
            'leitura': < leitura >}}

        :raise InvalidParameterError: Group User identifier is none or invalid.
        :raise GrupoUsuarioNaoExisteError: Group User not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_ugroup):
            raise InvalidParameterError(
                u'The identifier of Group User is invalid or was not informed.')

        url = 'ugroup/get/' + str(id_ugroup) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)