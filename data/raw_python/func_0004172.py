def validate_ipv6(self, id_vlan):
        """Validates ACL - IPv6 of VLAN from its identifier.

        Assigns 1 to 'acl_valida_v6'.

        :param id_vlan: Identifier of the Vlan. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: Vlan identifier is null and invalid.
        :raise VlanNaoExisteError: Vlan not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'The identifier of Vlan is invalid or was not informed.')

        url = 'vlan/' + str(id_vlan) + '/validate/' + IP_VERSION.IPv6[0] + '/'

        code, xml = self.submit(None, 'PUT', url)

        return self.response(code, xml)