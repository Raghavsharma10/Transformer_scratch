def _setup_frontend(self):
        ''' Receiver sockets facing clients (DAQ systems)
        '''
        self.frontends = []
        self.fe_poller = zmq.Poller()
        for actual_frontend_address in self.frontend_address:
            # Subscriber or server socket
            actual_frontend = (actual_frontend_address,
                               self.context.socket(self.frontend_socket_type))
            # Wait 0.5 s before termating socket
            actual_frontend[1].setsockopt(zmq.LINGER, 500)
            # Buffer only 10 meassages, then throw data away
            actual_frontend[1].set_hwm(10)
            # A suscriber has to set to not filter any data
            if self.frontend_socket_type == zmq.SUB:
                actual_frontend[1].setsockopt_string(zmq.SUBSCRIBE, u'')
            actual_frontend[1].connect(actual_frontend_address)
            self.frontends.append(actual_frontend)
            self.fe_poller.register(actual_frontend[1], zmq.POLLIN)
        self.raw_data = queue.Queue()
        self.fe_stop = threading.Event()