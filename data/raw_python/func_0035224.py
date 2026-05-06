def send_message(self, msg):
        """
        Internal method used to send messages through Clementine remote network protocol.
        """

        if self.socket is not None:

            msg.version = self.PROTOCOL_VERSION
            serialized = msg.SerializeToString()
            data = struct.pack(">I", len(serialized)) + serialized

            #print("Sending message: %s" % msg)
            try:
                self.socket.send(data)
            except Exception as e:
                #self.state = "Disconnected"
                pass