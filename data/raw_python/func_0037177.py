def remove(self, source_id, auth_ids):
        """ Remove one or more sets of authorization credentials from a Managed Source

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourceauthremove

            :param source_id: target Source ID
            :type source_id: str
            :param resources: An array of the authorization credential set IDs that you would like to remove.
            :type resources: array of str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': source_id, 'auth_ids': auth_ids}
        return self.request.post('remove', params)