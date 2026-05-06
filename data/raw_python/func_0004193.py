def edit_by_id(
            self,
            id_equip_acesso,
            id_tipo_acesso,
            fqdn,
            user,
            password,
            enable_pass):
        """Edit access type, fqdn, user, password and enable_pass of the relationship of
        equipment and access type.

        :param id_tipo_acesso: Access type identifier.
        :param id_equip_acesso: Equipment identifier.
        :param fqdn: Equipment FQDN.
        :param user: User.
        :param password: Password.
        :param enable_pass: Enable access.

        :return: None

        :raise InvalidParameterError: The parameters fqdn, user, password or
            access type id are invalid or none.
        :raise EquipamentoAcessoNaoExisteError: Equipment access type relationship doesn't exist.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_tipo_acesso):
            raise InvalidParameterError(
                u'Access type id is invalid or not informed.')

        equipamento_acesso_map = dict()
        equipamento_acesso_map['fqdn'] = fqdn
        equipamento_acesso_map['user'] = user
        equipamento_acesso_map['pass'] = password
        equipamento_acesso_map['enable_pass'] = enable_pass
        equipamento_acesso_map['id_tipo_acesso'] = id_tipo_acesso
        equipamento_acesso_map['id_equip_acesso'] = id_equip_acesso

        url = 'equipamentoacesso/edit/'

        code, xml = self.submit(
            {'equipamento_acesso': equipamento_acesso_map}, 'POST', url)

        return self.response(code, xml)