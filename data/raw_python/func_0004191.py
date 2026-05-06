def list_by_equip(self, name):
        """
        List all equipment access by equipment name

        :return: Dictionary with the following structure:

        ::

            {‘equipamento_acesso’:[ {'id': <id_equiptos_access>,
            'equipamento': <id_equip>,
            'fqdn': <fqdn>,
            'user': <user>,
            'password': <pass>
            'tipo_acesso': <id_tipo_acesso>,
            'enable_pass': <enable_pass> }]}

        :raise InvalidValueError: Invalid parameter.
        :raise EquipamentoNotFoundError: Equipment name not found in database.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        equip_access_map = dict()
        equip_access_map['name'] = name

        code, xml = self.submit(
            {"equipamento_acesso": equip_access_map}, 'POST', 'equipamentoacesso/name/')

        key = 'equipamento_acesso'
        return get_list_map(self.response(code, xml, [key]), key)