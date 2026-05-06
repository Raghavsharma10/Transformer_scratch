def get_related_environment_list(self, environment_vip_id):
        """Get all Environment by Environment Vip.

        :return: Following dictionary:

        ::

            {'ambiente': [{ 'id': <id_environment>,
            'grupo_l3': <id_group_l3>,
            'grupo_l3_name': <name_group_l3>,
            'ambiente_logico': <id_logical_environment>,
            'ambiente_logico_name': <name_ambiente_logico>,
            'divisao_dc': <id_dc_division>,
            'divisao_dc_name': <name_divisao_dc>,
            'filter': <id_filter>,
            'filter_name': <filter_name>,
            'link': <link> }, ... ]}


        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise DataBaseError: Can't connect to networkapi database.
        :raise XMLError: Failed to generate the XML response.
        """

        url = 'environment/environmentvip/{}/'.format(environment_vip_id)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml, ['environment_related_list'])