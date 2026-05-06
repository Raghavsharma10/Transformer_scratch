def alter(
            self,
            id_environment_vip,
            finalidade_txt,
            cliente_txt,
            ambiente_p44_txt,
            description):
        """Change Environment VIP from by the identifier.

        :param id_environment_vip: Identifier of the Environment VIP. Integer value and greater than zero.
        :param finalidade_txt: Finality. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param cliente_txt: ID  Client. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param ambiente_p44_txt: Environment P44. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: None

        :raise InvalidParameterError: Environment VIP identifier is null and invalid.
        :raise InvalidParameterError: The value of finalidade_txt, cliente_txt or ambiente_p44_txt is invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_environment_vip):
            raise InvalidParameterError(
                u'The identifier of Environment VIP is invalid or was not informed.')

        environmentvip_map = dict()
        environmentvip_map['finalidade_txt'] = finalidade_txt
        environmentvip_map['cliente_txt'] = cliente_txt
        environmentvip_map['ambiente_p44_txt'] = ambiente_p44_txt
        environmentvip_map['description'] = description

        url = 'environmentvip/' + str(id_environment_vip) + '/'

        code, xml = self.submit(
            {'environment_vip': environmentvip_map}, 'PUT', url)

        return self.response(code, xml)