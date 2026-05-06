def check_number_available(self, id_environment, num_vlan, id_vlan):
        """Checking if environment has a number vlan available

        :param id_environment: Identifier of environment
        :param num_vlan: Vlan number
        :param id_vlan: Vlan indentifier (False if inserting a vlan)

        :return: True is has number available, False if hasn't

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise InvalidParameterError: Invalid ID for VLAN.
        :raise VlanNaoExisteError: VLAN not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'vlan/check_number_available/' + \
            str(id_environment) + '/' + str(num_vlan) + '/' + str(id_vlan)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)