def associate_environment_option_pool(self, id_option_pool, id_environment):
        """Create a relationship of optionpool with Environment.

        :param id_option_pool: Identifier of the Option Pool. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment . Integer value and greater than zero.
        :return: Dictionary with the following structure:


            {‘id’: < id >,
                option: {
                    'id': <id>
                    'type':<type>
                    'name':<name> }
                environment: {
                    'id':<id>
                    .... all environment info }
                }

        :raise InvalidParameterError: Option Pool/Environment Pool identifier is null and/or invalid.
        :raise optionpoolNotFoundError: Option Pool not registered.
        :raise EnvironmentVipNotFoundError: Environment Pool not registered.
        :raise optionpoolError: Option Pool is already associated with the environment pool.
        :raise UserNotAuthorizedError: User does not have authorization to make this association.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_pool):
            raise InvalidParameterError(
                u'The identifier of Option Pool is invalid or was not informed.')

        if not is_valid_int_param(id_environment):
            raise InvalidParameterError(
                u'The identifier of Environment Pool is invalid or was not informed.')



        url= 'api/pools/environment_options/save/'

        return self.post(url,  {'option_id': id_option_pool,"environment_id":id_environment })