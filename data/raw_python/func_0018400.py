def __call(self, uri, params=None, method="get"):
        """Only returns the response, nor the status_code
        """
        try:
            resp = self.__get_response(uri, params, method, False)
            rjson = resp.json(**self.json_options)
            assert resp.ok
        except AssertionError:
            msg = "OCode-{}: {}".format(resp.status_code, rjson["message"])
            raise BadRequest(msg)
        except Exception as e:
            msg = "Bad response: {}".format(e)
            log.error(msg, exc_info=True)
            raise BadRequest(msg)
        else:
            return rjson