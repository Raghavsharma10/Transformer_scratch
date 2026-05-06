def connect(self, force=False):
        '''Establish a connection'''
        # Don't re-establish existing connections
        if not force and self.alive():
            return True

        self._reset()

        # Otherwise, try to connect
        with self._socket_lock:
            try:
                logger.info('Creating socket...')
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self._timeout)
                logger.info('Connecting to %s, %s', self.host, self.port)
                self._socket.connect((self.host, self.port))
                # Set our socket's blocking state to whatever ours is
                self._socket.setblocking(self._blocking)
                # Safely write our magic
                self._pending.append(constants.MAGIC_V2)
                while self.pending():
                    self.flush()
                # And send our identify command
                self.identify(self._identify_options)
                while self.pending():
                    self.flush()
                self._reconnnection_counter.success()
                # Wait until we've gotten a response to IDENTIFY, try to read
                # one. Also, only spend up to the provided timeout waiting to
                # establish the connection.
                limit = time.time() + self._timeout
                responses = self._read(1)
                while (not responses) and (time.time() < limit):
                    responses = self._read(1)
                if not responses:
                    raise ConnectionTimeoutException(
                        'Read identify response timed out (%ss)' % self._timeout)
                self.identified(responses[0])
                return True
            except:
                logger.exception('Failed to connect')
                if self._socket:
                    self._socket.close()
                self._reconnnection_counter.failed()
                self._reset()
                return False