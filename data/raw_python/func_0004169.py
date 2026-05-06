def confirm_vlan(self, number_net, id_environment_vlan, ip_version=None):
        """Checking if the vlan insert need to be confirmed

        :param number_net: Filter by vlan number column
        :param id_environment_vlan: Filter by environment ID related
        :param ip_version: Ip version for checking

        :return: True is need confirmation, False if no need

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise InvalidParameterError: Invalid ID for VLAN.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'vlan/confirm/' + \
            str(number_net) + '/' + id_environment_vlan + '/' + str(ip_version)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)