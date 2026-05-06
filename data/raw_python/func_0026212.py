def write(self, message):
        """
        Buffers each message until a newline is reached.  Each complete line is
        then published to the logging system through ``self.log()``.
        """

        self.__thread_local_ctx.write_count += 1

        try:
            if self.__thread_local_ctx.write_count > 1:
                return

            # For each line in the buffer ending with \n, output that line to
            # the logger
            msgs = (self.buffer + message).split('\n')
            self.buffer = msgs.pop(-1)
            for m in msgs:
                self.log_orig(m, echo=True)
        finally:
            self.__thread_local_ctx.write_count -= 1