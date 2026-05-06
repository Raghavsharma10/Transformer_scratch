def recv(self):
        """Receives a message from PS and decrypts it and returns a Message"""
        LOGGER.debug('Receiving')
        try:
            message_length = struct.unpack('>i', self._socket.recv(4))[0]
            message_length -= Connection.COMM_LENGTH
            LOGGER.debug('Length: %i', message_length)
        except socket.timeout:
            return None
        
        comm_status = struct.unpack('>i', self._socket.recv(4))[0]
        LOGGER.debug('Status: %i', comm_status)
        bytes_received = 0
        message = b""
        
        while bytes_received < message_length:
            if message_length - bytes_received >= 1024:
                recv_len = 1024
            else:
                recv_len = message_length - bytes_received
            bytes_received += recv_len
            LOGGER.debug('Received %i', bytes_received)
            message += self._socket.recv(recv_len)
        
        if comm_status == 0:
            message = self._crypt.decrypt(message)
        else:
            return Message(len(message), Connection.COMM_ERROR, message)
        
        msg = Message(message_length, comm_status, message)

        return msg