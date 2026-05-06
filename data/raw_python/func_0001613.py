def _set_timeouts(self, timeouts):
        """ Set socket timeouts for send and receive respectively """

        (send_timeout, recv_timeout) = (None, None)

        try:
            (send_timeout, recv_timeout) = timeouts
        except TypeError:
            raise EndpointError(
                '`timeouts` must be a pair of numbers (2, 3) which represent '
                'the timeout values for send and receive respectively')

        if send_timeout is not None:
            self.socket.set_int_option(
                nanomsg.SOL_SOCKET, nanomsg.SNDTIMEO, send_timeout)

        if recv_timeout is not None:
            self.socket.set_int_option(
                nanomsg.SOL_SOCKET, nanomsg.RCVTIMEO, recv_timeout)