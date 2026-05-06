def send_connection_request(self):
        """
        Sends a ConnectionRequest to the iDigi server using the credentials
        established with the id of the monitor as defined in the monitor
        member.
        """
        try:
            self.log.info("Sending ConnectionRequest for Monitor %s."
                          % self.monitor_id)
            # Send connection request and perform a receive to ensure
            # request is authenticated.
            # Protocol Version = 1.
            payload = struct.pack('!H', 0x01)
            # Username Length.
            payload += struct.pack('!H', len(self.client.username))
            # Username.
            payload += six.b(self.client.username)
            # Password Length.
            payload += struct.pack('!H', len(self.client.password))
            # Password.
            payload += six.b(self.client.password)
            # Monitor ID.
            payload += struct.pack('!L', int(self.monitor_id))

            # Header 6 Bytes : Type [2 bytes] & Length [4 Bytes]
            # ConnectionRequest is Type 0x01.
            data = struct.pack("!HL", CONNECTION_REQUEST, len(payload))

            # The full payload.
            data += payload

            # Send Connection Request.
            self.socket.send(data)

            # Set a 60 second blocking on recv, if we don't get any data
            # within 60 seconds, timeout which will throw an exception.
            self.socket.settimeout(60)

            # Should receive 10 bytes with ConnectionResponse.
            response = self.socket.recv(10)

            # Make socket blocking.
            self.socket.settimeout(0)

            if len(response) != 10:
                raise PushException("Length of Connection Request Response "
                                    "(%d) is not 10." % len(response))

            # Type
            response_type = int(struct.unpack("!H", response[0:2])[0])
            if response_type != CONNECTION_RESPONSE:
                raise PushException(
                    "Connection Response Type (%d) is not "
                    "ConnectionResponse Type (%d)." % (response_type, CONNECTION_RESPONSE))

            status_code = struct.unpack("!H", response[6:8])[0]
            self.log.info("Got ConnectionResponse for Monitor %s. Status %s."
                          % (self.monitor_id, status_code))
            if status_code != STATUS_OK:
                raise PushException("Connection Response Status Code (%d) is "
                                    "not STATUS_OK (%d)." % (status_code, STATUS_OK))
        except Exception as exception:
            # TODO(posborne): This is bad!  It isn't necessarily a socket exception!
            # Likely a socket exception, close it and raise an exception.
            self.socket.close()
            self.socket = None
            raise exception