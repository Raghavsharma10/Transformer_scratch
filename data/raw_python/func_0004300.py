def search(self, **kwargs):
        """
        Method to search object group permissions general based on extends search.

        :param search: Dict containing QuerySets to find object group permissions general.
        :param include: Array containing fields to include on response.
        :param exclude: Array containing fields to exclude on response.
        :param fields:  Array containing fields to override default fields.
        :param kind: Determine if result will be detailed ('detail') or basic ('basic').
        :return: Dict containing object group permissions general
        """

        return super(ApiObjectGroupPermissionGeneral, self).get(self.prepare_url('api/v3/object-group-perm-general/',
                                                                                 kwargs))