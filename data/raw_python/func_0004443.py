def get_all_environment_option_pool(self, id_environment=None, option_id=None, option_type=None):
        """Get all Option VIP by Environment .

        :return: Dictionary with the following structure:

        ::

            {[{‘id’: < id >,
                option: {
                    'id': <id>
                    'type':<type>
                    'name':<name> }
                environment: {
                    'id':<id>
                    .... all environment info }
                    etc to option pools ...] }

        :raise EnvironmentVipNotFoundError: Environment Pool not registered.
        :raise DataBaseError: Can't connect to networkapi database.
        :raise XMLError: Failed to generate the XML response.
        """
        url='api/pools/environment_options/'

        if id_environment:
            if  option_id:
                if option_type:
                    url = url + "?environment_id=" + str(id_environment)+ "&option_id=" + str(option_id)  + "&option_type=" + option_type
                else:
                    url = url + "?environment_id=" + str(id_environment)+ "&option_id=" + str(option_id)
            else:
                if option_type:
                    url = url + "?environment_id=" + str(id_environment) + "&option_type=" + option_type
                else:
                    url = url + "?environment_id=" + str(id_environment)
        elif option_id:
            if option_type:
                url = url + "?option_id=" + str(option_id)  + "&option_type=" + option_type
            else:
                url = url + "?option_id=" + str(option_id)
        elif option_type:
            url = url + "?option_type=" + option_type


        return self.get(url)