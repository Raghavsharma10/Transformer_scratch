def create(self, environments):
        """
        Method to create environments

        :param environments: Dict containing environments desired to be created on database
        :return: None
        """

        data = {'environments': environments}
        return super(ApiEnvironment, self).post('api/v3/environment/', data)