def update(self, ogpgs):
        """
        Method to update object group permissions general

        :param ogpgs: List containing object group permissions general desired to updated
        :return: None
        """

        data = {'ogpgs': ogpgs}
        ogpgs_ids = [str(ogpg.get('id')) for ogpg in ogpgs]

        return super(ApiObjectGroupPermissionGeneral, self).put('api/v3/object-group-perm-general/%s/' %
                                                                ';'.join(ogpgs_ids), data)