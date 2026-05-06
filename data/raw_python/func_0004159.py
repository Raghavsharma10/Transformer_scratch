def edit_vlan(
            self,
            environment_id,
            name,
            number,
            description,
            acl_file,
            acl_file_v6,
            id_vlan):
        """Edit a VLAN

        :param id_vlan: ID for Vlan
        :param environment_id: ID for Environment.
        :param name: The name of VLAN.
        :param description: Some description to VLAN.
        :param number: Number of Vlan
        :param acl_file: Acl IPv4 File name to VLAN.
        :param acl_file_v6: Acl IPv6 File name to VLAN.

        :return: None

        :raise VlanError: VLAN name already exists, DC division of the environment invalid or there is no VLAN number available.
        :raise VlanNaoExisteError: VLAN not found.
        :raise AmbienteNaoExisteError: Environment not registered.
        :raise InvalidParameterError: Name of Vlan and/or the identifier of the Environment is null or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'Vlan id is invalid or was not informed.')

        if not is_valid_int_param(environment_id):
            raise InvalidParameterError(u'Environment id is none or invalid.')

        if not is_valid_int_param(number):
            raise InvalidParameterError(u'Vlan number is none or invalid')

        vlan_map = dict()
        vlan_map['vlan_id'] = id_vlan
        vlan_map['environment_id'] = environment_id
        vlan_map['name'] = name
        vlan_map['description'] = description
        vlan_map['acl_file'] = acl_file
        vlan_map['acl_file_v6'] = acl_file_v6
        vlan_map['number'] = number

        code, xml = self.submit({'vlan': vlan_map}, 'POST', 'vlan/edit/')

        return self.response(code, xml)