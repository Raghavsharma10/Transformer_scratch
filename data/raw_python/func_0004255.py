def delete_environment(self, environment_ids):
        """
        Method to delete environment

        :param environment_ids: Ids of Environment
        """
        uri = 'api/v3/environment/%s/' % environment_ids

        return super(ApiEnvironment, self).delete(uri)