def get(self, id, service='facebook', type='analysis'):
        """ Get a given Pylon task

            :param id: The ID of the task
            :type id: str
            :param service: The PYLON service (facebook)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """
        return self.request.get(service + '/task/' + type + '/' + id)