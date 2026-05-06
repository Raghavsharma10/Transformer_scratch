def setup(self):
        '''Called after instantiation'''
        TelnetHandlerBase.setup(self)
        # Spawn a greenlet to handle socket input
        self.greenlet_ic = gevent.spawn(self.inputcooker)
        # Note that inputcooker exits on EOF
        
        # Sleep for 0.5 second to allow options negotiation
        gevent.sleep(0.5)