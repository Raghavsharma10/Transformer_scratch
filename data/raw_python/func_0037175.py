def remove(self, source_id, resource_ids):
        """ Remove one or more resources from a Managed Source

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourceresourceremove

            :param source_id: target Source ID
            :type source_id: str
            :param resources: An array of the resource IDs that you would like to remove..
            :type resources: array of str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': source_id, 'resource_ids': resource_ids}
        return self.request.post('remove', params)