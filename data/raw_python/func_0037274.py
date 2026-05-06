def compile(self, csdl):
        """ Compile the given CSDL.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/compile

            Raises a DataSiftApiException for any error given by the REST API, including CSDL compilation.

            :param csdl: CSDL to compile
            :type csdl: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('compile', data=dict(csdl=csdl))