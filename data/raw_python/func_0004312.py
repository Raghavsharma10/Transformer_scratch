def buscar_timeout_opcvip(self, id_ambiente_vip):
        """Buscar nome_opcao_txt das Opcoes VIp quando tipo_opcao = 'Timeout' pelo environmentvip_id

        :return: Dictionary with the following structure:

        ::
            {‘timeout_opt’: ‘timeout_opt’: <'nome_opcao_txt'>}

        :raise InvalidParameterError: Environment VIP identifier is null and invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise InvalidParameterError: finalidade_txt and cliente_txt is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_ambiente_vip):
            raise InvalidParameterError(
                u'The identifier of environment-vip is invalid or was not informed.')

        url = 'environment-vip/get/timeout/' + str(id_ambiente_vip) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)