def _reset(self):
        '''Reset all of our stateful variables'''
        self._socket = None
        # The pending messages we have to send, and the current buffer we're
        # sending
        self._pending = deque()
        self._out_buffer = ''
        # Our read buffer
        self._buffer = ''
        # The identify response we last received from the server
        self._identify_response = {}
        # Our ready state
        self.last_ready_sent = 0
        self.ready = 0