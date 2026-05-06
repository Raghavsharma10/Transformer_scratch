def update_environment(self, environment, environment_ids):
        """
        Method to update environment

        :param environment_ids: Ids of Environment
        """

        uri = 'api/v3/environment/%s/' % environment_ids

        data = dict()
        data['environments'] = list()
        data['environments'].append(environment)

        return super(ApiEnvironment, self).put(uri, data)