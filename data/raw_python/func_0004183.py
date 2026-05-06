def update(self, neighbors):
        """
        Method to update neighbors

        :param neighbors: List containing neighbors desired to updated
        :return: None
        """

        data = {'neighbors': neighbors}
        neighbors_ids = [str(env.get('id')) for env in neighbors]

        return super(ApiV4Neighbor, self).put('api/v4/neighbor/%s/' %
                                               ';'.join(neighbors_ids), data)