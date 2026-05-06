def disassociate_environment_option_pool(self, environment_option_id):
        """Remove a relationship of optionpool with Environment.

        :param id_option_pool: Identifier of the Option Pool. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment Pool. Integer value and greater than zero.

        :return: { 'id': < environment_option_id> }

        :raise InvalidParameterError: Option Pool/Environment Pool identifier is null and/or invalid.
        :raise optionpoolNotFoundError: Option Pool not registered.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise optionpoolError: Option pool is not associated with the environment pool
        :raise UserNotAuthorizedError: User does not have authorization to make this association.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(environment_option_id):
            raise InvalidParameterError(
                u'The identifier of Option Pool is invalid or was not informed.')

        if not is_valid_int_param(environment_option_id):
            raise InvalidParameterError(
                u'The identifier of Environment Pool is invalid or was not informed.')

        url = 'api/pools/environment_options/' + str(environment_option_id) +  '/'

        return self.delete(url)