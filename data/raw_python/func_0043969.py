def contents(self):
        """
        Returns entire contents of course page

        :returns: response object

        Example Usage::

        >>> import muddle
        >>> muddle.course(10).content()
        """

        params = self.request_params
        params.update({'wsfunction': 'core_course_get_contents',
                       'courseid': self.course_id})

        return requests.get(self.api_url, params=params, verify=False).json()