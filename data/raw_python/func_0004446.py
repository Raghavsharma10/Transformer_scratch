def modify_environment_option_pool(self, environment_option_id, id_option_pool,id_environment ):
        """Remove a relationship of optionpool with Environment.

        :param id_option_pool: Identifier of the Option Pool. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment Pool. Integer value and greater than zero.

                :return: Dictionary with the following structure:

        ::

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
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise optionpoolError: Option pool is not associated with the environment pool
        :raise UserNotAuthorizedError: User does not have authorization to make this association.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(environment_option_id):
            raise InvalidParameterError(
                u'The identifier of Environment Option Pool is invalid or was not informed.')

        #optionpool_map = dict()
        #optionpool_map['option'] = option_id
        #optionpool_map['environment'] = environment_id


        url = 'api/pools/environment_options/' + str(environment_option_id) +  '/'

        return self.put(url, {'option_id': id_option_pool,"environment_id":id_environment })