def create_queue(self, vhost, name, **kwargs):
        """
        Create a queue. The API documentation specifies that all of the body
        elements are optional, so this method only requires arguments needed
        to form the URI

        :param string vhost: The vhost to create the queue in.
        :param string name: The name of the queue

        More on these operations can be found at:
        http://www.rabbitmq.com/amqp-0-9-1-reference.html

        """

        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['queues_by_name'] % (vhost, name)

        body = json.dumps(kwargs)

        return self._call(path,
                                 'PUT',
                                 body,
                                 headers=Client.json_headers)