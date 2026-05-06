def delete(self, ids):
        """
        Method to delete object group permissions by their ids

        :param ids: Identifiers of object group permissions
        :return: None
        """
        url = build_uri_with_ids('api/v3/object-group-perm/%s/', ids)
        return super(ApiObjectGroupPermission, self).delete(url)