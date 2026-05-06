def alterar(self, id_tipo_acesso, protocolo):
        """Edit access type by its identifier.

        :param id_tipo_acesso: Access type identifier.
        :param protocolo: Protocol.

        :return: None

        :raise ProtocoloTipoAcessoDuplicadoError: Protocol already exists.
        :raise InvalidParameterError: Protocol value is invalid or none.
        :raise TipoAcessoNaoExisteError: Access type doesn't exist.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_tipo_acesso):
            raise InvalidParameterError(
                u'Access type id is invalid or was not informed.')

        tipo_acesso_map = dict()
        tipo_acesso_map['protocolo'] = protocolo

        url = 'tipoacesso/' + str(id_tipo_acesso) + '/'

        code, xml = self.submit({'tipo_acesso': tipo_acesso_map}, 'PUT', url)

        return self.response(code, xml)