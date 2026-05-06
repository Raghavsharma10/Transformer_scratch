def associate(self, id_option_vip, id_environment_vip):
        """Create a relationship of OptionVip with EnvironmentVip.

        :param id_option_vip: Identifier of the Option VIP. Integer value and greater than zero.
        :param id_environment_vip: Identifier of the Environment VIP. Integer value and greater than zero.

        :return: Following dictionary

        ::

            {'opcoesvip_ambiente_xref': {'id': < id_opcoesvip_ambiente_xref >} }

        :raise InvalidParameterError: Option VIP/Environment VIP identifier is null and/or invalid.
        :raise OptionVipNotFoundError: Option VIP not registered.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise OptionVipError: Option vip is already associated with the environment vip.
        :raise UserNotAuthorizedError: User does not have authorization to make this association.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_vip):
            raise InvalidParameterError(
                u'The identifier of Option VIP is invalid or was not informed.')

        if not is_valid_int_param(id_environment_vip):
            raise InvalidParameterError(
                u'The identifier of Environment VIP is invalid or was not informed.')

        url = 'optionvip/' + \
            str(id_option_vip) + '/environmentvip/' + str(id_environment_vip) + '/'

        code, xml = self.submit(None, 'PUT', url)

        return self.response(code, xml)