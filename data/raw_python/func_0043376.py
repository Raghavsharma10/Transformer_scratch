def set_gateway(self, gateway):
        '''
        :param crabpy.gateway.crab.CrabGateway gateway: Gateway to use.
        '''
        self.gateway = gateway
        self.gewest.gateway = gateway