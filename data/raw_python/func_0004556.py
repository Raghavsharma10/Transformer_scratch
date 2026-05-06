def update(self, environments):
        """
        Method to update environments vip

        :param environments vip: List containing environments vip desired
                                 to updated
        :return: None
        """

        data = {'environments_vip': environments}
        environments_ids = [str(env.get('id')) for env in environments]

        uri = 'api/v3/environment-vip/%s/' % ';'.join(environments_ids)
        return super(ApiEnvironmentVip, self).put(uri, data)