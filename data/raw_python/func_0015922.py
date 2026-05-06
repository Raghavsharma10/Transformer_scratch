def get_queue_bindings(self, vhost, qname):
        """
        Return a list of dicts, one dict per binding. The dict format coming
        from RabbitMQ for queue named 'testq' is:

        {"source":"sourceExch","vhost":"/","destination":"testq",
         "destination_type":"queue","routing_key":"*.*","arguments":{},
         "properties_key":"%2A.%2A"}
        """
        vhost = quote(vhost, '')
        qname = quote(qname, '')
        path = Client.urls['bindings_on_queue'] % (vhost, qname)
        bindings = self._call(path, 'GET')
        return bindings