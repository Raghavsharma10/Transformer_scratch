def create_exchange(self,
                        vhost,
                        name,
                        xtype,
                        auto_delete=False,
                        durable=True,
                        internal=False,
                        arguments=None):
        """
        Creates an exchange in the given vhost with the given name. As per the
        RabbitMQ API documentation, a JSON body also needs to be included that
        "looks something like this":

        {"type":"direct",
        "auto_delete":false,
        "durable":true,
        "internal":false,
        "arguments":[]}

        On success, the API returns a 204 with no content, in which case this
        function returns True. If any other response is received, it's raised.

        :param string vhost: Vhost to create the exchange in.
        :param string name: Name of the proposed exchange.
        :param string type: The AMQP exchange type.
        :param bool auto_delete: Whether or not the exchange should be
            dropped when the no. of consumers drops to zero.
        :param bool durable: Whether you want this exchange to persist a
            broker restart.
        :param bool internal: Whether or not this is a queue for use by the
            broker only.
        :param list arguments: If given, should be a list. If not given, an
            empty list is sent.

        """

        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['exchange_by_name'] % (vhost, name)
        base_body = {"type": xtype, "auto_delete": auto_delete,
                     "durable": durable, "internal": internal,
                     "arguments": arguments or list()}

        body = json.dumps(base_body)
        self._call(path, 'PUT', body,
                          headers=Client.json_headers)
        return True