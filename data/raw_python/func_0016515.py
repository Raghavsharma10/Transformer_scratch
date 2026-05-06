def publish(self, message, exchange, routing_key, mandatory=None,
            immediate=None, headers=None):
        """Publish a message to a named exchange."""
        body, properties = message

        if headers:
            properties.headers = headers

        ret = self.channel.basic_publish(body=body,
                                         properties=properties,
                                         exchange=exchange,
                                         routing_key=routing_key,
                                         mandatory=mandatory,
                                         immediate=immediate)
        if mandatory or immediate:
            self.close()