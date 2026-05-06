def __error_middleware(self, res, res_json):
        """
        Middleware that raises an exception when HTTP statuscode is an error code.
        """
        if(res.status_code in [400, 401, 402, 403, 404, 405, 406, 409]):
            err_dict = res_json.get('error', {})
            raise UpCloudAPIError(error_code=err_dict.get('error_code'),
                                  error_message=err_dict.get('error_message'))

        return res_json