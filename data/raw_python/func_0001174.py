def get_brokers(self):
        """
        Parses the KAKFA_URL and returns a list of hostname:port pairs in the format
        that kafka-python expects.
        """
        return ['{}:{}'.format(parsedUrl.hostname, parsedUrl.port) for parsedUrl in
                [urlparse(url) for url in self.kafka_url.split(',')]]