def remove(self, id_vlan):
        """Remove a VLAN by your primary key.
        Execute script to remove VLAN

        :param id_vlan: ID for VLAN.

        :return: Following dictionary:

        ::

          {‘sucesso’: {‘codigo’: < codigo >,
          ‘descricao’: {'stdout':< stdout >, 'stderr':< stderr >}}}

        :raise InvalidParameterError: Identifier of the VLAN is invalid.
        :raise VlanNaoExisteError: VLAN not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'Parameter id_vlan is invalid. Value: ' +
                id_vlan)

        url = 'vlan/' + str(id_vlan) + '/remove/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)