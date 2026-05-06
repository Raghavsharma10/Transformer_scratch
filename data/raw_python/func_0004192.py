def inserir(
            self,
            id_equipamento,
            fqdn,
            user,
            password,
            id_tipo_acesso,
            enable_pass):
        """Add new relationship between equipment and access type and returns its id.

        :param id_equipamento: Equipment identifier.
        :param fqdn: Equipment FQDN.
        :param user: User.
        :param password: Password.
        :param id_tipo_acesso: Access Type identifier.
        :param enable_pass: Enable access.

        :return: Dictionary with the following: {‘equipamento_acesso’: {‘id’: < id >}}

        :raise EquipamentoNaoExisteError: Equipment doesn't exist.
        :raise TipoAcessoNaoExisteError: Access Type doesn't exist.
        :raise EquipamentoAcessoError:  Equipment and access type already associated.
        :raise InvalidParameterError: The parameters equipment id, fqdn, user, password or
            access type id are invalid or none.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        equipamento_acesso_map = dict()
        equipamento_acesso_map['id_equipamento'] = id_equipamento
        equipamento_acesso_map['fqdn'] = fqdn
        equipamento_acesso_map['user'] = user
        equipamento_acesso_map['pass'] = password
        equipamento_acesso_map['id_tipo_acesso'] = id_tipo_acesso
        equipamento_acesso_map['enable_pass'] = enable_pass

        code, xml = self.submit(
            {'equipamento_acesso': equipamento_acesso_map}, 'POST', 'equipamentoacesso/')

        return self.response(code, xml)