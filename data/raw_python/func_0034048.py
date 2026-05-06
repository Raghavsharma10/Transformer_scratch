def on_message(self, msg=None):
        """
        Poll the websocket for a new packet.

        `Client.listen()` calls this.

        :param msg (string(byte array)): Optional. Parse the specified message
            instead of receiving a packet from the socket.
        """
        if msg is None:
            try:
                msg = self.ws.recv()
            except Exception as e:
                self.subscriber.on_message_error(
                    'Error while receiving packet: %s' % str(e))
                self.disconnect()
                return False

        if not msg:
            self.subscriber.on_message_error('Empty message received')
            return False

        buf = BufferStruct(msg)
        opcode = buf.pop_uint8()
        try:
            packet_name = packet_s2c[opcode]
        except KeyError:
            self.subscriber.on_message_error('Unknown packet %s' % opcode)
            return False

        if not self.ingame and packet_name in ingame_packets:
            self.subscriber.on_ingame()
            self.ingame = True

        parser = getattr(self, 'parse_%s' % packet_name)
        try:
            parser(buf)
        except BufferUnderflowError as e:
            msg = 'Parsing %s packet failed: %s' % (packet_name, e.args[0])
            self.subscriber.on_message_error(msg)

        if len(buf.buffer) != 0:
            msg = 'Buffer not empty after parsing "%s" packet' % packet_name
            self.subscriber.on_message_error(msg)

        return packet_name