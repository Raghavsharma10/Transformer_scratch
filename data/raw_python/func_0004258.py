def update(self, environments):
        """
        Method to update environments

        :param environments: List containing environments desired to updated
        :return: None
        """

        data = {'environments': environments}
        environments_ids = [str(env.get('id')) for env in environments]

        return super(ApiEnvironment, self).put('api/v3/environment/%s/' %
                                               ';'.join(environments_ids), data)