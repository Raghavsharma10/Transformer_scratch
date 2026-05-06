def inserir(self, id_permission, read, write, id_group):
        """Inserts a new Administrative Permission and returns its identifier.

        :param id_permission: Identifier of the Permission. Integer value and greater than zero.
        :param read: Read. 0 or 1
        :param write: Write. 0 or 1
        :param id_group: Identifier of the Group of User. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'perm': {'id': < id_perm >}}

        :raise InvalidParameterError: The identifier of Administrative Permission, identifier of Group of User, read or  write is null and invalid.
        :raise ValorIndicacaoPermissaoInvalidoError: The value of read or write is null and invalid.
        :raise PermissaoAdministrativaDuplicadaError: Function already registered for the user group.
        :raise GrupoUsuarioNaoExisteError: Group of User not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        perms_map = dict()
        perms_map['id_permission'] = id_permission
        perms_map['read'] = read
        perms_map['write'] = write
        perms_map['id_group'] = id_group

        code, xml = self.submit(
            {'administrative_permission': perms_map}, 'POST', 'aperms/')

        return self.response(code, xml)