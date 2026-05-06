def sign(self, method, params):
        """Calculate signature with the SIG_METHOD(HMAC-SHA1)
        Returns a base64 encoeded string of the hex signature

        :param method: the http verb
        :param params: the params needs calculate
        """
        query_str = utils.percent_encode(params.items(), True)

        str_to_sign = "{0}&%2F&{1}".format(
            method, utils.percent_quote(query_str)
        )

        sig = hmac.new(
            utils.to_bytes(self._secret_key + "&"),
            utils.to_bytes(str_to_sign),
            hashlib.sha1
        )
        return base64.b64encode(sig.digest())