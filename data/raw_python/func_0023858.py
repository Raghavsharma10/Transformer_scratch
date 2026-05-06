def attach_ip(self, server, family='IPv4'):
        """
        Attach a new (random) IPAddress to the given server (object or UUID).
        """
        body = {
            'ip_address': {
                'server': str(server),
                'family': family
            }
        }

        res = self.request('POST', '/ip_address', body)
        return IPAddress(cloud_manager=self, **res['ip_address'])