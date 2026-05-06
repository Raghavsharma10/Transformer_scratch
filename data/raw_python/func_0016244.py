def set(self, address, host_name):
        """
        Assign one or more PTR record to a single IP Address
        :type address: str
        :type host_name: list[str]
        :param address: (str) The IP address to configure
        :param host_name: (list[str]) The list of strings representing PTR records
        :return: (bool) True in case of success, False in case of failure
        """
        request = self._call(SetEnqueueSetReverseDns.SetEnqueueSetReverseDns, IP=address, Hosts=host_name)
        response = request.commit()
        return response['Success']