def buscar_healthchecks(self, id_ambiente_vip):
        """Search healthcheck ​by environmentvip_id

        :return: Dictionary with the following structure:

        ::

            {'healthcheck_opt': [{'name': <name>, 'id': <id>},...]}

        :raise InvalidParameterError: Environment VIP identifier is null and invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise InvalidParameterError: id_ambiente_vip is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'environment-vip/get/healthcheck/' + str(id_ambiente_vip)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml, ['healthcheck_opt'])