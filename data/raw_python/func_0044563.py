def expected_signature(self):
        """
        The AWS SigV4 signature expected from the request.
        """
        k_secret = b"AWS4" + self.key_mapping[self.access_key].encode("utf-8")
        k_date = hmac.new(k_secret, self.request_date.encode("utf-8"),
                          sha256).digest()
        k_region = hmac.new(k_date, self.region.encode("utf-8"),
                            sha256).digest()
        k_service = hmac.new(k_region, self.service.encode("utf-8"),
                             sha256).digest()
        k_signing = hmac.new(k_service, _aws4_request_bytes, sha256).digest()

        return hmac.new(k_signing, self.string_to_sign.encode("utf-8"),
                        sha256).hexdigest()