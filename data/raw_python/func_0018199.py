def setup(self):
        '''Called after instantiation'''
        TelnetHandlerBase.setup(self)
        # Spawn a greenlet to handle socket input
        self.greenlet_ic = eventlet.spawn(self.inputcooker)
        # Note that inputcooker exits on EOF

        # Sleep for 0.5 second to allow options negotiation
        eventlet.sleep(0.5)