def create(self, label, status=None, master=None):
        """ Create an Identity

            :param label: The label to give this new identity
            :param status: The status of this identity. Default: 'active'
            :param master: Represents whether this identity is a master.
                Default: False
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'label': label}

        if status:
            params['status'] = status
        if master:
            params['master'] = master

        return self.request.post('', params)