def list(self, per_page=None, page=None, status=None, service='facebook'):
        """ Get a list of Pylon tasks

            :param per_page: How many tasks to display per page
            :type per_page: int
            :param page: Which page of tasks to display
            :type page: int
            :param status: The status of the tasks to list
            :type page: string
            :param service: The PYLON service (facebook)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {}

        if per_page is not None:
            params['per_page'] = per_page
        if page is not None:
            params['page'] = page
        if status:
            params['status'] = status

        return self.request.get(service + '/task', params)