def stop(self, id, service='facebook'):
        """ Stop the recording for the provided id

            :param id: The hash to start recording with
            :type id: str
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """
        return self.request.post(service + '/stop', data=dict(id=id))