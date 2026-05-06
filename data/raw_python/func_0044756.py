def receive_data(self):  # pragma: no cover; no covered since qt event loop
        ''' Infinite loop via QObject.moveToThread(), does not block event loop
        '''
        while(not self._stop_readout.wait(0.01)):  # use wait(), do not block
            if self._send_data:
                if self.socket_type != zmq.DEALER:
                    raise RuntimeError('You send data without a bidirectional '
                                       'connection! Define a bidirectional '
                                       'connection.')
                self.receiver.send(self._send_data)
                self._send_data = None
            try:
                data_serialized = self.receiver.recv(flags=zmq.NOBLOCK)
                data = self.deserializer(data_serialized)
                self.data.emit(data)
            except zmq.Again:
                pass
        self.finished.emit()