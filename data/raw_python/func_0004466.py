def alterar(self, id_perm, id_permission, read, write, id_group):
        """Change Administrative Permission from by the identifier.

        :param id_perm: Identifier of the Administrative Permission. Integer value and greater than zero.
        :param id_permission: Identifier of the Permission. Integer value and greater than zero.
        :param read: Read. 0 or 1
        :param write: Write. 0 or 1
        :param id_group: Identifier of the Group of User. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Administrative Permission, identifier of Permission, identifier of Group of User, read or  write is null and invalid.
        :raise ValorIndicacaoPermissaoInvalidoError: The value of read or write is null and invalid.
        :raise PermissaoAdministrativaNaoExisteError: Administrative Permission not registered.
        :raise GrupoUsuarioNaoExisteError: Group of User not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_perm):
            raise InvalidParameterError(
                u'The identifier of Administrative Permission is invalid or was not informed.')

        url = 'aperms/' + str(id_perm) + '/'

        perms_map = dict()
        perms_map['id_perm'] = id_perm
        perms_map['id_permission'] = id_permission
        perms_map['read'] = read
        perms_map['write'] = write
        perms_map['id_group'] = id_group

        code, xml = self.submit(
            {'administrative_permission': perms_map}, 'PUT', url)

        return self.response(code, xml)