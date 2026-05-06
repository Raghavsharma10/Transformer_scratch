def create(self, equipments):
        """
        Method to create equipments

        :param equipments: List containing equipments desired to be created on database
        :return: None
        """

        data = {'equipments': equipments}
        return super(ApiEquipment, self).post('api/v3/equipment/', data)