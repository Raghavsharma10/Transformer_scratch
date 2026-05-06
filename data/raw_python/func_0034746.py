def do_async_send(self):
        """
        Send any queued data. This function should only be called after a write
        event on a file descriptor.
        """
        assert len(self.sendbuf)

        nwritten = self.sock.send(self.sendbuf)
        nframes = 0

        for entry in self.sendbuf_frames:
            frame, offset, callback = entry

            if offset <= nwritten:
                nframes += 1

                if callback:
                    callback()
            else:
                entry[1] -= nwritten

        self.sendbuf = self.sendbuf[nwritten:]
        self.sendbuf_frames = self.sendbuf_frames[nframes:]