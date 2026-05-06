def clientConnectionFailed(self, connector, reason):
        """Connection failed
        """
        print('Connection failed. Reason:', reason)        
        ReconnectingClientFactory.clientConnectionFailed(self, connector, reason)