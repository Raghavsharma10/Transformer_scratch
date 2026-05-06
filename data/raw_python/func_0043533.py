def set_gateway(self, gateway):
        '''
        :param crabpy.gateway.capakey.CapakeyGateway gateway: Gateway to use.
        '''
        self.gateway = gateway
        if (self._gemeente is not None):
            self._gemeente.set_gateway(gateway)