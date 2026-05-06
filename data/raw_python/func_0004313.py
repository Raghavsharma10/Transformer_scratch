def buscar_rules(self, id_ambiente_vip, id_vip=''):
        """Search rules ​by environmentvip_id

        :return: Dictionary with the following structure:

        ::

            {'name_rule_opt': [{'name_rule_opt': <name>, 'id': <id>}, ...]}

        :raise InvalidParameterError: Environment VIP identifier is null and invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise InvalidParameterError: finalidade_txt and cliente_txt is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'environment-vip/get/rules/' + \
            str(id_ambiente_vip) + '/' + str(id_vip)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)