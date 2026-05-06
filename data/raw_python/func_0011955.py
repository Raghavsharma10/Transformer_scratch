def close_cache(self):
        '''
        Close cache of WS Shinken
        '''
        # Close all WS_Shinken cache files
        for server in self.servers:
            if self.servers[server]['cache'] == True:
                self.servers[server]['file'].close()