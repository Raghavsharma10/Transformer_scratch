def request_signature(self):
        """
        The signature passed in the request.
        """
        signature = self.query_parameters.get(_x_amz_signature)
        if signature is not None:
            signature = signature[0]
        else:
            signature = self.authorization_header_parameters.get(_signature)
            if signature is None:
                raise AttributeError("Signature was not passed in the request")
            
        return signature