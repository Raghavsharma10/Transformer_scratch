def delete(self, ids):
        """
        Method to delete object group permissions general by their ids

        :param ids: Identifiers of object group permissions general
        :return: None
        """
        url = build_uri_with_ids('api/v3/object-group-perm-general/%s/', ids)
        return super(ApiObjectGroupPermissionGeneral, self).delete(url)