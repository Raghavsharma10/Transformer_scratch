def create(self, neighbors):
        """
        Method to create neighbors

        :param neighbors: List containing neighbors desired to be created on database
        :return: None
        """

        data = {'neighbors': neighbors}
        return super(ApiV4Neighbor, self).post('api/v4/neighbor/', data)