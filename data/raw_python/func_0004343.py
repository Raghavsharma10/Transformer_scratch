def remover(self, id_tipo_acesso):
        """Removes access type by its identifier.

        :param id_tipo_acesso: Access type identifier.

        :return: None

        :raise TipoAcessoError: Access type associated with equipment, cannot be removed.
        :raise InvalidParameterError: Protocol value is invalid or none.
        :raise TipoAcessoNaoExisteError: Access type doesn't exist.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_tipo_acesso):
            raise InvalidParameterError(
                u'Access type id is invalid or was not informed.')

        url = 'tipoacesso/' + str(id_tipo_acesso) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)