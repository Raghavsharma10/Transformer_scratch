def post(self, url, obj, content_type=JSON_CONTENT_TYPE, **kwargs):
        """
        POST an object and check the response. Retry once if a badNonce error
        is received.

        :param str url: The URL to request.
        :param ~josepy.interfaces.JSONDeSerializable obj: The serializable
            payload of the request.
        :param bytes content_type: The expected content type of the response.
            By default, JSON.

        :raises txacme.client.ServerError: If server response body carries HTTP
            Problem (draft-ietf-appsawg-http-problem-00).
        :raises acme.errors.ClientError: In case of other protocol errors.
        """
        def retry_bad_nonce(f):
            f.trap(ServerError)
            # The current RFC draft defines the namespace as
            # urn:ietf:params:acme:error:<code>, but earlier drafts (and some
            # current implementations) use urn:acme:error:<code> instead. We
            # don't really care about the namespace here, just the error code.
            if f.value.message.typ.split(':')[-1] == 'badNonce':
                # If one nonce is bad, others likely are too. Let's clear them
                # and re-add the one we just got.
                self._nonces.clear()
                self._add_nonce(f.value.response)
                return self._post(url, obj, content_type, **kwargs)
            return f
        return (
            self._post(url, obj, content_type, **kwargs)
            .addErrback(retry_bad_nonce))