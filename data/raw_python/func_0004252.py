def get_environment(self, environment_ids):
        """
        Method to get environment
        """

        uri = 'api/v3/environment/%s/' % environment_ids

        return super(ApiEnvironment, self).get(uri)