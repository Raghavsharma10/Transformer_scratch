def _commit(self):
        """
        :return: (dict) Response object content
        """
        assert self.uri is not None, Exception("BadArgument: uri property cannot be None")
        url = '{}/{}'.format(self.uri, self.__class__.__name__)
        serialized_json = jsonpickle.encode(self, unpicklable=False, )
        headers = {'Content-Type': 'application/json', 'Content-Length': str(len(serialized_json))}
        response = Http.post(url=url, data=serialized_json, headers=headers)
        if response.status_code != 200:
            from ArubaCloud.base.Errors import MalformedJsonRequest
            raise MalformedJsonRequest("Request: {}, Status Code: {}".format(serialized_json, response.status_code))
        content = jsonpickle.decode(response.content.decode("utf-8"))
        if content['ResultCode'] == 17:
            from ArubaCloud.base.Errors import OperationAlreadyEnqueued
            raise OperationAlreadyEnqueued("{} already enqueued".format(self.__class__.__name__))
        if content['Success'] is False:
            from ArubaCloud.base.Errors import RequestFailed
            raise RequestFailed("Request: {}, Response: {}".format(serialized_json, response.content))
        return content