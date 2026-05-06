def associate(self, environment_id, environment_vip_id):

        """Associate a news Environment on Environment VIP and returns its identifier.

        :param environment_id: Identifier of the Environment. Integer value and greater than zero.
        :param environment_vip_id: Identifier of the Environment VIP. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {'environment_environment_vip': {'id': < id >}}

        :raise InvalidParameterError: The value of environment_id or environment_vip_id is invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(environment_id):
            raise InvalidParameterError(
                u'The identifier of Environment VIP is invalid or was not informed.')

        if not is_valid_int_param(environment_vip_id):
            raise InvalidParameterError(
                u'The identifier of Environment is invalid or was not informed.')

        environment_environment_vip_map = dict()
        environment_environment_vip_map['environment_id'] = environment_id
        environment_environment_vip_map['environment_vip_id'] = environment_vip_id

        url = 'environment/{}/environmentvip/{}/'.format(environment_id, environment_vip_id)

        code, xml = self.submit(None, 'PUT', url)

        return self.response(code, xml)