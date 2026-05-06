def queue_send(self, frame, callback=None, recv_callback=None):
        """
        Enqueue `frame` to the send buffer so that it is send on the next
        `do_async_send`. `callback` is an optional callable to call when the
        frame has been fully written. `recv_callback` is an optional callable
        to quickly set the `recv_callback` attribute to.
        """
        frame = self.apply_send_hooks(frame, False)
        self.sendbuf += frame.pack()
        self.sendbuf_frames.append([frame, len(self.sendbuf), callback])

        if recv_callback:
            self.recv_callback = recv_callback