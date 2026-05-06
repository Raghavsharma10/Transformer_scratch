def query_bytes(self, transport, num_bytes, header, *data):
        """Queries for binary data

        :param  transport: A transport object.
        :param num_bytes: The exact number of data bytes expected.
        :param header: The message header.
        :param data: Optional data.
        :returns: The raw unparsed data bytearray.

        """
        message = self.create_message(header, *data)
        logger.debug('SignalRecovery query bytes: %r', message)
        with transport:
            transport.write(message)

            response = transport.read_exactly(num_bytes)
            logger.debug('SignalRecovery response: %r', response)
            # We need to read 3 bytes, because there is a \0 character
            # separating the data from the status bytes.
            _, status_byte, overload_byte = transport.read_exactly(3)

        logger.debug('SignalRecovery stb: %r olb: %r', status_byte, overload_byte)
        self.call_byte_handler(status_byte, overload_byte)
        # returns raw unparsed bytes.
        return response