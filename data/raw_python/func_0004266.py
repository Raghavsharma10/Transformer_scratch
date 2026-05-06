def delete(self, ids):
        """
        Method to delete equipments by their id's

        :param ids: Identifiers of equipments
        :return: None
        """
        url = build_uri_with_ids('api/v4/equipment/%s/', ids)
        return super(ApiV4Equipment, self).delete(url)