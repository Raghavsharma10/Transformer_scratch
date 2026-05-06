def gen_def_json_scheme(self, req, method_fields=None):
        """
        Generate the scheme for the json request.
        :param req: String representing the name of the method to call
        :param method_fields: A dictionary containing the method-specified fields
        :rtype : json object representing the method call
        """
        json_dict = dict(
            ApplicationId=req,
            RequestId=req,
            SessionId=req,
            Password=self.auth.password,
            Username=self.auth.username
        )
        if method_fields is not None:
            json_dict.update(method_fields)
        self.logger.debug(json.dumps(json_dict))
        return json.dumps(json_dict)