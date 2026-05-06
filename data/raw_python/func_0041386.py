def check_assert(self, cmd, retries=1, pollrate=60, on_retry=None):
        """
        Run a command, logging (using gather) and raise an exception if the
        return code of the command indicates failure.
        Try the command multiple times if requested.

        :param cmd <string|list>: A shell command
        :param retries int: The number of times to try before declaring failure
        :param pollrate int: how long to sleep between tries
        :param on_retry <string|list>: A shell command to run before retrying a failure
        :return: (stdout,stderr) if exit code is zero
        """

        for try_num in range(0, retries):
            if try_num > 0:
                self.logger.debug(
                    "assert: Failed {} times. Retrying in {} seconds: {}".
                    format(try_num, pollrate, cmd))
                time.sleep(pollrate)
                if on_retry is not None:
                    self.gather(on_retry)  # no real use for the result though

            result, stdout, stderr = self.gather(cmd)
            if result == SUCCESS:
                break

        self.logger.debug("assert: Final result = {} in {} tries.".format(result, try_num))

        assertion.success(
            result,
            "Error running [{}] {}.\n{}".
            format(pushd.Dir.getcwd(), cmd, stderr))

        return stdout, stderr