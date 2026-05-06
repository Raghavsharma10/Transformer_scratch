def create(self, asns):
        """
        Method to create asns

        :param asns: List containing asns desired to be created on database
        :return: None
        """

        data = {'asns': asns}
        return super(ApiV4As, self).post('api/v4/as/', data)