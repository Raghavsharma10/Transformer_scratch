def details(self):
        """
        Returns details for given category

        :returns: category response object

        Example Usage::

        >>> import muddle
        >>> muddle.category(10).details()
        """
        params = {'wsfunction': 'core_course_get_categories',
                  'criteria[0][key]': 'id',
                  'criteria[0][value]': self.category_id}

        params.update(self.request_params)

        return requests.post(self.api_url, params=params, verify=False)