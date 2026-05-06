def _passes_filter(self, proxy):


        ''' avoid redudant and space consuming calls to 'self' '''
        
        ''' validate proxy based on provided filters '''
        if self.allowed_countries is not None and proxy['country'] not in self.allowed_countries:
            return False
        if self.denied_countries is not None and  proxy['country'] in self.denied_countries:
            return False
        if self.https_only and proxy['https'] == False:
            return False

        if not self.all_ports and str(proxy.port) not in self.ports:
            return False

        return True