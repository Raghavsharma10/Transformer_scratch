def update(self, interfaces=None):
        """
        Method to update interface.
        :param interfaces: List containing interface's desired to be updated on database.
        :return: None.
        """

        data = {'interfaces': interfaces}

        return super(ApiInterfaceRequest, self).put('api/v3/interface/', data)