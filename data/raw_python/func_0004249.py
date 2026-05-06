def inserir(self, name, read, write, edit, remove):
        """Insert new user group and returns its identifier.

        :param name: User group's name.
        :param read: If user group has read permission ('S' ou 'N').
        :param write: If user group has write permission ('S' ou 'N').
        :param edit: If user group has edit permission ('S' ou 'N').
        :param remove: If user group has remove permission ('S' ou 'N').

        :return: Dictionary with structure: {'user_group': {'id': < id >}}

        :raise InvalidParameterError: At least one of the parameters is invalid or none..
        :raise NomeGrupoUsuarioDuplicadoError: User group name already exists.
        :raise ValorIndicacaoPermissaoInvalidoError: Read, write, edit or remove value is invalid.
        :raise DataBaseError: Networkapi failed to access database.
        :raise XMLError: Networkapi fails generating response XML.
        """
        ugroup_map = dict()
        ugroup_map['nome'] = name
        ugroup_map['leitura'] = read
        ugroup_map['escrita'] = write
        ugroup_map['edicao'] = edit
        ugroup_map['exclusao'] = remove

        code, xml = self.submit({'user_group': ugroup_map}, 'POST', 'ugroup/')

        return self.response(code, xml)