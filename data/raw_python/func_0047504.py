def build_items(self):
        u"""This method called by Executer.
        /proc/net/tcp -> {host:host, key:key, value:value, clock:clock}
        """
        protocols = ['tcp', 'tcp6']
        for protocol in protocols:
            procfile = open('/proc/net/{0}'.format(protocol), 'r')
            stats = self.count(procfile)

            for key, value in stats.items():
                item = NetstatItem(key=key,
                                   value=value,
                                   host=self.hostname
                                   )

                self.queue.put(item, block=False)