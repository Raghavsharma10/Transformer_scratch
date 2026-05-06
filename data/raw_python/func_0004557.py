def create(self, environments):
        """
        Method to create environments vip

        :param environments vip: Dict containing environments vip desired
                                 to be created on database
        :return: None
        """

        data = {'environments_vip': environments}
        uri = 'api/v3/environment-vip/'
        return super(ApiEnvironmentVip, self).post(uri, data)