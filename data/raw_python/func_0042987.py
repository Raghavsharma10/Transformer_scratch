def get_signature(self, req):
        """calculate the signature of the oss request
        Returns the signatue
        """
        oss_url = url.URL(req.url)

        oss_headers = [
            "{0}:{1}\n".format(key, val)
            for key, val in req.headers.lower_items()
            if key.startswith(self.X_OSS_PREFIX)
        ]
        canonicalized_headers = "".join(sorted(oss_headers))
        logger.debug(
            "canonicalized header : [{0}]".format(canonicalized_headers)
        )

        oss_url.params = {
            key: val
            for key, val in oss_url.params.items()
            if key in self.SUB_RESOURCES or key in self.OVERRIDE_QUERIES
        }

        oss_url.forge(key=lambda x: x[0])
        canonicalized_str = "{0}/{1}{2}".format(
            canonicalized_headers,
            self.get_bucket(oss_url.host),
            oss_url.uri
        )

        str_to_sign = "\n".join([
            req.method,
            req.headers["content-md5"],
            req.headers["content-type"],
            req.headers["date"],
            canonicalized_str
        ])
        logger.debug(
            "signature str is \n{0}\n{1}\n{0}\n".format("#" * 20, str_to_sign)
        )
        if isinstance(str_to_sign, requests.compat.str):
            str_to_sign = str_to_sign.encode("utf8")

        signature_bin = hmac.new(self._secret_key, str_to_sign, hashlib.sha1)
        signature = base64.b64encode(signature_bin.digest()).decode("utf8")
        logger.debug("signature is [{0}]".format(signature))
        return signature