def get_provincie_by_id(self, niscode):
        '''
        Retrieve a `provincie` by the niscode.

        :param integer niscode: The niscode of the provincie.
        :rtype: :class:`Provincie`
        '''
        def creator():
            for p in self.provincies:
                if p[0] == niscode:
                    return Provincie(p[0], p[1], Gewest(p[2]))

        if self.caches['permanent'].is_configured:
            key = 'GetProvincieByProvincieNiscode#%s' % niscode
            provincie = self.caches['permanent'].get_or_create(key, creator)
        else:
            provincie = creator()
        if provincie == None:
            raise GatewayResourceNotFoundException()
        provincie.set_gateway(self)
        return provincie