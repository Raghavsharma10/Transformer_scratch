def listPrimaryDSTypes(self, **kwargs):
        """
        API to list primary dataset types

        :param primary_ds_type: List that primary dataset type (Optional)
        :type primary_ds_type: str
        :param dataset: List the primary dataset type for that dataset (Optional)
        :type dataset: str
        :returns: List of dictionaries containing the following keys (primary_ds_type_id, data_type)
        :rtype: list of dicts

        """
        validParameters = ['primary_ds_type', 'dataset']

        checkInputParameter(method="listPrimaryDSTypes", parameters=kwargs.keys(), validParameters=validParameters)

        return self.__callServer("primarydstypes", params=kwargs)