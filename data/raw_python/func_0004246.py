def update(self, virtual_interfaces):
        """
        Method to update Virtual Interfaces

        :param Virtual Interfaces: List containing Virtual Interfaces desired to updated
        :return: None
        """

        data = {'virtual_interfaces': virtual_interfaces}
        virtual_interfaces_ids = [str(env.get('id')) for env in virtual_interfaces]

        return super(ApiV4VirtualInterface, self).put\
            ('api/v4/virtual-interface/%s/' % ';'.join(virtual_interfaces_ids), data)