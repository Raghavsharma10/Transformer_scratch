def dpu(self, hash=None, historics_id=None):
        """ Calculate the DPU cost of consuming a stream.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/dpu

            :param hash: target CSDL filter hash
            :type hash: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        if hash:
            return self.request.get('dpu', params=dict(hash=hash))
        if historics_id:
            return self.request.get('dpu', params=dict(historics_id=historics_id))