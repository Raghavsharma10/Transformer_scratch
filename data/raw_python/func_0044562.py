def string_to_sign(self):
        """
        The AWS SigV4 string being signed.
        """
        return (AWS4_HMAC_SHA256 + "\n" +
                self.request_timestamp + "\n" +
                self.credential_scope + "\n" +
                sha256(self.canonical_request.encode("utf-8")).hexdigest())