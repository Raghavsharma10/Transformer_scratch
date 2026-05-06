def update(self, equipments):
        """
        Method to update equipments

        :param equipments: List containing equipments desired to updated
        :return: None
        """

        data = {'equipments': equipments}
        equipments_ids = [str(env.get('id')) for env in equipments]

        return super(ApiV4Equipment, self).put('api/v4/equipment/%s/' %
                                               ';'.join(equipments_ids), data)