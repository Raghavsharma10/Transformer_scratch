def create(self, virtual_interfaces):
        """
        Method to create Virtual Interfaces

        :param Virtual Interfaces: List containing Virtual Interfaces desired to be created on database
        :return: None
        """

        data = {'virtual_interfaces': virtual_interfaces}
        return super(ApiV4VirtualInterface, self).post\
            ('api/v4/virtual-interface/', data)