def validate(self, signature, timestamp, nonce):
        """Validate request signature.

        :param signature: A string signature parameter sent by weixin.
        :param timestamp: A int timestamp parameter sent by weixin.
        :param nonce: A int nonce parameter sent by weixin.
        """
        if not self.token:
            raise RuntimeError('WEIXIN_TOKEN is missing')

        if self.expires_in:
            try:
                timestamp = int(timestamp)
            except (ValueError, TypeError):
                # fake timestamp
                return False

            delta = time.time() - timestamp
            if delta < 0:
                # this is a fake timestamp
                return False

            if delta > self.expires_in:
                # expired timestamp
                return False

        values = [self.token, str(timestamp), str(nonce)]
        s = ''.join(sorted(values))
        hsh = hashlib.sha1(s.encode('utf-8')).hexdigest()
        return signature == hsh