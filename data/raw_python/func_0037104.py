def validate(self, csdl, service='facebook'):
        """ Validate the given CSDL

            :param csdl: The CSDL to be validated for analysis
            :type csdl: str
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('validate', data=dict(csdl=csdl))