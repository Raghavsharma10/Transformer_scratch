def wait_for(self, pattern, timeout=None):
        """
        Block until a pattern have been found in stdout and stderr

        Args:
            pattern(:class:`~re.Pattern`): The pattern to search
            timeout(int): Maximum number of second to wait. If None, wait infinitely

        Raises: 
            TimeoutError: When timeout is reach
        """
        should_continue = True

        if self.block:
            raise TypeError(NON_BLOCKING_ERROR_MESSAGE)

        def stop(signum, frame):  # pylint: disable=W0613
            nonlocal should_continue
            if should_continue:
                raise TimeoutError()

        if timeout:
            signal.signal(signal.SIGALRM, stop)
            signal.alarm(timeout)

        while should_continue:
            output = self.poll_output() + self.poll_error()
            filtered = [line for line in output if re.match(pattern, line)]
            if filtered:
                should_continue = False