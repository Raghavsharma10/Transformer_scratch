def create(self, ogps):
        """
        Method to create object group permissions

        :param ogps: List containing vrf desired to be created on database
        :return: None
        """

        data = {'ogps': ogps}
        return super(ApiObjectGroupPermission, self).post('api/v3/object-group-perm/', data)