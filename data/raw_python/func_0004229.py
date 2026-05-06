def connecting_interfaces(self, interfaces):
        """
        Method to connecting interfaces.
        :param interfaces: List containing a dictionary with the interfaces ids and front or back.
        :return: 200 OK.
        """

        data = {'interfaces': interfaces}

        url = 'api/v3/connections/' + str(interfaces[0].get('id')) + '/' + str(interfaces[1].get('id')) + '/'

        return super(ApiInterfaceRequest, self).post(url, data)