def wait(self, duration=None, count=0):
        """
        Publish publishes the data argument to the given subject.

        Args:
            duration (float): will wait for the given number of seconds
            count (count): stop of wait after n messages from any subject
        """
        start = time.time()
        total = 0
        while True:
            type, result = self._recv(MSG, PING, OK)
            if type is MSG:
                total += 1
                if self._handle_msg(result) is False:
                    break

                if count and total >= count:
                    break

            elif type is PING:
                self._handle_ping()

            if duration and time.time() - start > duration:
                break