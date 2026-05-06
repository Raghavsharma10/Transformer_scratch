def create(self, ogpgs):
        """
        Method to create object group permissions general

        :param ogpgs: List containing vrf desired to be created on database
        :return: None
        """

        data = {'ogpgs': ogpgs}
        return super(ApiObjectGroupPermissionGeneral, self).post('api/v3/object-group-perm-general/', data)