def _generate_presence_token(self, channel_name):
        """Generate a presence token.

        :param str channel_name: Name of the channel to generate a signature for.
        :rtype: str
        """
        subject = "{}:{}:{}".format(self.connection.socket_id, channel_name, json.dumps(self.user_data))
        h = hmac.new(self.secret_as_bytes, subject.encode('utf-8'), hashlib.sha256)
        auth_key = "{}:{}".format(self.key, h.hexdigest())

        return auth_key