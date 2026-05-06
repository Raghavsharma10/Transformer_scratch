def merged(self):
        '''The clean stats from all the hosts reporting to this host.'''
        stats = {}
        for topic in self.client.topics()['topics']:
            for producer in self.client.lookup(topic)['producers']:
                hostname = producer['broadcast_address']
                port = producer['http_port']
                host = '%s_%s' % (hostname, port)
                stats[host] = nsqd.Client(
                    'http://%s:%s/' % (hostname, port)).clean_stats()
        return stats