def invoke(self, ns, request_name, params={}, simplify=False):
        """
        Invokes zimbra method using established authentication session.
        @param req: zimbra request
        @parm params: request params
        @param simplify: True to return python object, False to return xml struct
        @return: zimbra response
        @raise AuthException: if authentication fails
        @raise SoapException: wrapped server exception
        """
        if self.auth_token == None:
            raise AuthException('Unable to invoke zimbra method')

        if util.empty(request_name):
            raise ZimbraClientException('Invalid request')

        return self.transport.invoke(ns,
                                     request_name,
                                     params,
                                     self.auth_token,
                                     simplify)