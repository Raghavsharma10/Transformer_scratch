def publish(self, vhost, xname, rt_key, payload, payload_enc='string',
                properties=None):
        """
        Publish a message to an exchange.

        :param string vhost: vhost housing the target exchange
        :param string xname: name of the target exchange
        :param string rt_key: routing key for message
        :param string payload: the message body for publishing
        :param string payload_enc: encoding of the payload. The only choices
                      here are 'string' and 'base64'.
        :param dict properties: a dict of message properties
        :returns: boolean indicating success or failure.
        """
        vhost = quote(vhost, '')
        xname = quote(xname, '')
        path = Client.urls['publish_to_exchange'] % (vhost, xname)
        body = json.dumps({'routing_key': rt_key, 'payload': payload,
                           'payload_encoding': payload_enc,
                           'properties': properties or {}})
        result = self._call(path, 'POST', body)
        return result['routed']