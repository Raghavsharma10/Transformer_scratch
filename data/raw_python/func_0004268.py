def create(self, equipments):
        """
        Method to create equipments

        :param equipments: List containing equipments desired to be created on database
        :return: None
        """

        data = {'equipments': equipments}
        return super(ApiV4Equipment, self).post('api/v4/equipment/', data)