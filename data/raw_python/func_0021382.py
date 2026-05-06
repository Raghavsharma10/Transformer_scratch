def _resolve(self):
        """
        Query the consul DNS server for the service IP and port
        """
        endpoints = {}
        r = self.resolver.query(self.service, 'SRV')
        for rec in r.response.additional:
            name = rec.name.to_text()
            addr = rec.items[0].address
            endpoints[name] = {'addr': addr}
        for rec in r.response.answer[0].items:
            name = '.'.join(rec.target.labels)
            endpoints[name]['port'] = rec.port
        return [
            'http://{ip}:{port}'.format(
                ip=v['addr'], port=v['port']
            ) for v in endpoints.values()
        ]