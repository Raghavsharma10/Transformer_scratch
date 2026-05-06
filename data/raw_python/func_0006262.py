def connect(self):
        """
            Open a new telnet session on the remote server.
        """
        self.client = BaitTelnetClient(self.options['server'], self.options['port'])
        self.client.set_option_negotiation_callback(self.process_options)