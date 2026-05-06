def reset(self, addresses):
        """
        Remove all PTR records from the given address
        :type addresses: List[str]
        :param addresses: (List[str]) The IP Address to reset
        :return: (bool) True in case of success, False in case of failure
        """
        request = self._call(SetEnqueueResetReverseDns.SetEnqueueResetReverseDns, IPs=addresses)
        response = request.commit()
        return response['Success']