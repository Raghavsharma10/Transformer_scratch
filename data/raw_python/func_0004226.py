def create(self, interface):
        """
        Method to add an interface.
        :param interface: List containing interface's desired to be created on database.
        :return: Id.
        """

        data = {'interfaces': interface}
        return super(ApiInterfaceRequest, self).post('api/v3/interface/', data)