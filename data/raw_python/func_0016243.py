def get(self, addresses):
        """
        :type addresses: list[str]
        :param addresses: (list[str]) List of addresses to retrieve their reverse dns
        Retrieve the current configured ReverseDns entries
        :return: (list) List containing the current ReverseDns Addresses
        """
        request = self._call(GetReverseDns.GetReverseDns, IPs=addresses)
        response = request.commit()
        return response['Value']