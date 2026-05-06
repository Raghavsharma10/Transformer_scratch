def _generate_auth_token(self, channel_name):
        """Generate a token for authentication with the given channel.

        :param str channel_name: Name of the channel to generate a signature for.
        :rtype: str
        """
        subject = "{}:{}".format(self.connection.socket_id, channel_name)
        h = hmac.new(self.secret_as_bytes, subject.encode('utf-8'), hashlib.sha256)
        auth_key = "{}:{}".format(self.key, h.hexdigest())

        return auth_key