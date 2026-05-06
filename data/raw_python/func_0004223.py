def search(self, **kwargs):
        """
        Method to search interfaces based on extends search.
        :return: Dict containing interfaces.
        """

        return super(ApiInterfaceRequest, self).get(self.prepare_url('api/v3/interface/', kwargs))