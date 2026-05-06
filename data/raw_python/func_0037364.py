def validate(self, output_type, output_params):
        """ Check that a subscription is defined correctly.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushvalidate

            :param output_type:   One of DataSift's supported output types, e.g. s3
            :type output_type: str
            :param output_params: The set of parameters required by the specified output_type for docs on all available connectors see http://dev.datasift.com/docs/push/connectors/
            :type output_params: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('validate',
                                 dict(output_type=output_type, output_params=output_params))