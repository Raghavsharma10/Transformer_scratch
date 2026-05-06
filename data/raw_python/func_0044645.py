def _setup_backend(self):
        ''' Send sockets facing services (e.g. online monitor, other forwarders)
        '''
        self.backends = []
        self.be_poller = zmq.Poller()
        for actual_backend_address in self.backend_address:
            # publisher or client socket
            actual_backend = (actual_backend_address,
                              self.context.socket(self.backend_socket_type))
            # Wait 0.5 s before termating socket
            actual_backend[1].setsockopt(zmq.LINGER, 500)
            # Buffer only 100 meassages, then throw data away
            actual_backend[1].set_hwm(10)
            actual_backend[1].bind(actual_backend_address)
            self.backends.append(actual_backend)
            if self.backend_socket_type != zmq.DEALER:
                self.be_poller.register(actual_backend[1], zmq.POLLIN)
        self.be_stop = threading.Event()