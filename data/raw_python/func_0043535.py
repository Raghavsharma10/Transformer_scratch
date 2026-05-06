def set_gateway(self, gateway):
        '''
        :param crabpy.gateway.capakey.CapakeyGateway gateway: Gateway to use.
        '''
        self.gateway = gateway
        self.sectie.set_gateway(gateway)