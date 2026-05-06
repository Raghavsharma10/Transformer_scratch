def from_normal(self, hosts=default.ELASTICSEARCH_HOSTS, **kwargs):
        """
        Initialize a Elasticsearch client by specified hosts list.

        :param hosts: list of nodes we should connect to. Node should be a
            dictionary ({"host": "localhost", "port": 9200}), the entire dictionary
            will be passed to the :class:`~elasticsearch.Connection` class as
            kwargs, or a string in the format of ``host[:port]`` which will be
            translated to a dictionary automatically.  If no value is given the
            :class:`~elasticsearch.Urllib3HttpConnection` class defaults will be used

        :return: void
        """
        self.client = Elasticsearch(hosts=hosts, **kwargs)
        logger.info('Initialize normal Elasticsearch Client: %s.' % self.client)