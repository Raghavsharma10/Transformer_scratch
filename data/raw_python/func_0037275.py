def is_valid(self, csdl):
        """ Checks if the given CSDL is valid.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/validate

            :param csdl: CSDL to validate
            :type csdl: str
            :returns: Boolean indicating the validity of the CSDL
            :rtype: bool
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        try:
            self.validate(csdl)
        except DataSiftApiException as e:
            if e.response.status_code == 400:
                return False
            else:
                raise e
        return True