def setup(self):
        '''Called after instantiation'''
        TelnetHandlerBase.setup(self)
        # Spawn a thread to handle socket input
        self.thread_ic = threading.Thread(target=self.inputcooker)
        self.thread_ic.setDaemon(True)
        self.thread_ic.start()
        # Note that inputcooker exits on EOF
        
        # Sleep for 0.5 second to allow options negotiation
        time.sleep(0.5)