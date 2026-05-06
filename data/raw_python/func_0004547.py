def remove(self, id_environment_vip):
        """Remove Environment VIP from by the identifier.

        :param id_environment_vip: Identifier of the Environment VIP. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: Environment VIP identifier is null and invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise EnvironmentVipError: There networkIPv4 or networkIPv6 associated with environment vip.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_environment_vip):
            raise InvalidParameterError(
                u'The identifier of Environment VIP is invalid or was not informed.')

        url = 'environmentvip/' + str(id_environment_vip) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)