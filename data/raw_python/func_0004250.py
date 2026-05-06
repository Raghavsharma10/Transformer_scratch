def alterar(self, id_user_group, name, read, write, edit, remove):
        """Edit user group data from its identifier.

        :param id_user_group: User group id.
        :param name: User group name.
        :param read: If user group has read permission ('S' ou 'N').
        :param write: If user group has write permission ('S' ou 'N').
        :param edit: If user group has edit permission ('S' ou 'N').
        :param remove: If user group has remove permission ('S' ou 'N').

        :return: None

        :raise NomeGrupoUsuarioDuplicadoError: User group name already exists.
        :raise ValorIndicacaoPermissaoInvalidoError: Read, write, edit or remove value is invalid.
        :raise GrupoUsuarioNaoExisteError: User Group not found.
        :raise InvalidParameterError: At least one of the parameters is invalid or none.
        :raise DataBaseError: Networkapi failed to access database.
        :raise XMLError: Networkapi fails generating response XML.
        """
        if not is_valid_int_param(id_user_group):
            raise InvalidParameterError(
                u'Invalid or inexistent user group id.')

        url = 'ugroup/' + str(id_user_group) + '/'

        ugroup_map = dict()
        ugroup_map['nome'] = name
        ugroup_map['leitura'] = read
        ugroup_map['escrita'] = write
        ugroup_map['edicao'] = edit
        ugroup_map['exclusao'] = remove

        code, xml = self.submit({'user_group': ugroup_map}, 'PUT', url)

        return self.response(code, xml)