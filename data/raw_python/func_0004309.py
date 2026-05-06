def remove(self, id_option_vip):
        """Remove Option VIP from by the identifier.

        :param id_option_vip: Identifier of the Option VIP. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: Option VIP identifier is null and invalid.
        :raise OptionVipNotFoundError: Option VIP not registered.
        :raise OptionVipError: Option VIP  associated with environment vip.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_vip):
            raise InvalidParameterError(
                u'The identifier of Option VIP is invalid or was not informed.')

        url = 'optionvip/' + str(id_option_vip) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)