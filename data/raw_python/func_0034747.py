def do_async_recv(self, bufsize):
        """
        Receive any completed frames from the socket. This function should only
        be called after a read event on a file descriptor.
        """
        data = self.sock.recv(bufsize)

        if len(data) == 0:
            raise socket.error('no data to receive')

        self.recvbuf += data

        while contains_frame(self.recvbuf):
            frame, self.recvbuf = pop_frame(self.recvbuf)
            frame = self.apply_recv_hooks(frame, False)

            if not self.recv_callback:
                raise ValueError('no callback installed for %s' % frame)

            self.recv_callback(frame)