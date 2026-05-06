def inserir(self, protocolo):
        """Insert new access type and returns its identifier.

        :param protocolo: Protocol.

        :return: Dictionary with structure: {‘tipo_acesso’: {‘id’: < id >}}

        :raise ProtocoloTipoAcessoDuplicadoError: Protocol already exists.
        :raise InvalidParameterError: Protocol value is invalid or none.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        tipo_acesso_map = dict()
        tipo_acesso_map['protocolo'] = protocolo

        code, xml = self.submit(
            {'tipo_acesso': tipo_acesso_map}, 'POST', 'tipoacesso/')

        return self.response(code, xml)