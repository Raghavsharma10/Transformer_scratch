def request(self, service, data):
        """
        Makes a call to TinyLetter's __svcbus__ endpoint.
        """
        _res = self._request(service, data)
        res = _res.json()[0][0]
        if res["success"] == True:
            return res["result"]
        else:
            err_msg = res["errmsg"]
            raise Exception("Request not successful: '{0}'".format(err_msg))