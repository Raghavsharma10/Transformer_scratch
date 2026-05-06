def remove(self, id_option_pool):
        """Remove Option pool  by  identifier and all Environment related .

        :param id_option_pool: Identifier of the Option Pool. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: Option Pool identifier is null and invalid.
        :raise optionpoolNotFoundError: Option Pool not registered.
        :raise optionpoolError: Option Pool associated with Pool.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_pool):
            raise InvalidParameterError(
                u'The identifier of Option Pool is invalid or was not informed.')

        url = 'api/pools/options/' + str(id_option_pool) + '/'

        return self.delete(url)