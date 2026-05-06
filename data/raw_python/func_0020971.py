def unpack(self, message):
        """Called to extract a STOMP message into this instance.

        message:
            This is a text string representing a valid
            STOMP (v1.0) message.

        This method uses unpack_frame(...) to extract the
        information, before it is assigned internally.

        retuned:
            The result of the unpack_frame(...) call.

        """
        if not message:
            raise FrameError("Unpack error! The given message isn't valid '%s'!" % message)

        msg = unpack_frame(message)

        self.cmd = msg['cmd']
        self.headers = msg['headers']

        # Assign directly as the message will have the null
        # character in the message already.
        self.body = msg['body']

        return msg