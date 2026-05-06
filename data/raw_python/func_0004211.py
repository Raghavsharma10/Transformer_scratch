def update(self, ogps):
        """
        Method to update object group permissions

        :param ogps: List containing object group permissions desired to updated
        :return: None
        """

        data = {'ogps': ogps}
        ogps_ids = [str(ogp.get('id')) for ogp in ogps]

        return super(ApiObjectGroupPermission, self).put('api/v3/object-group-perm/%s/' %
                                                         ';'.join(ogps_ids), data)