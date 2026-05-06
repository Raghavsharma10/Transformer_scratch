def send(self, message, fragment_size=None, mask=False):
        """
        Send a message. If `fragment_size` is specified, the message is
        fragmented into multiple frames whose payload size does not extend
        `fragment_size`.
        """
        for frame in self.message_to_frames(message, fragment_size, mask):
            self.send_frame(frame)