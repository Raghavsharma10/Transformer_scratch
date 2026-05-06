def get_response_message(self, resp_msg_signature=None, delay=COMMAND_DELAY):
        """ Receive data from connected gateway and if required seach and return a stream that starts at the required
        response message signature.  The reason we couple the search for the response signature here is that given the
        RNET protocol and TCP comms, we dont have an easy way of knowign that we have received the response.  We want to
        minimise the time spent reading the socket (to reduce user lag), hence we use the message response signature
        at this point to determine when to stop reading."""

        matching_message = None  # Set intial value to none (assume no response found)
        if resp_msg_signature is None:
            no_of_socket_reads = 1  # If we are not looking for a specific response do a single read to clear the buffer
        else:
            no_of_socket_reads = 10 # Try 10x (= approx 1s at default)if we are looking for a specific response

        time.sleep(delay)  # Insert recommended delay to ensure command is processed correctly
        self.sock.setblocking(0)  # Needed to prevent request for waiting indefinitely

        data = B''
        for i in range(0, no_of_socket_reads):
            try:
                # Receive what has been sent
                data += self.sock.recv(4096)
                _LOGGER.debug('i= %s; len= %s data= %s', i, len(data), '[{}]'.format(', '.join(hex(x) for x in data)))
            except BlockingIOError:  # Expected outcome if there is not data
                _LOGGER.debug('Passed=%s', i)
                pass
            except ConnectionResetError as msg:
                _LOGGER.error("Error trying to connect to Russound controller. Check that no other device or system "
                              "is using the port that you are trying to connect to. "
                              "Try resetting the bridge you are using to connect.")
                _LOGGER.error(msg)
            # Check if we have our message.  If so break out else keep looping.
            if resp_msg_signature is not None:  # If we are looking for a specific response
                matching_message, data = self.find_signature(data, resp_msg_signature)
            if matching_message is not None:  # Required response found
                _LOGGER.debug("Number of reads=%s", i + 1)
                break
            time.sleep(delay)  # Wait before reading again - default of 100ms
        return matching_message