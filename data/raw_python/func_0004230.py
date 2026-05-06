def disconnecting_interfaces(self, interfaces, **kwargs):
        """
        Method to remove the link between interfaces.
        :param interfaces: List of ids.
        :return: 200 OK.
        """

        url = 'api/v3/connections/' + str(interfaces[0]) + '/' + str(interfaces[1]) + '/'

        return super(ApiInterfaceRequest, self).delete(self.prepare_url(url, kwargs))