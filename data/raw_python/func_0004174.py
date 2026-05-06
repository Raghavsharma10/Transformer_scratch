def deallocate(self, id_vlan):
        """Deallocate all relationships between Vlan.

        :param id_vlan: Identifier of the VLAN. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: VLAN identifier is null and invalid.
        :raise VlanError: VLAN is active.
        :raise VlanNaoExisteError: VLAN not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'The identifier of Vlan is invalid or was not informed.')

        url = 'vlan/' + str(id_vlan) + '/deallocate/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)