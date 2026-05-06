def run(self):
        """ Drain the process output streams. """
        read_stdout = partial(self._read_output, stream=self._process.stdout,
                              callback=self._callback_stdout,
                              output_file=self._stdout_file)

        read_stderr = partial(self._read_output, stream=self._process.stderr,
                              callback=self._callback_stderr,
                              output_file=self._stderr_file)

        # capture the process output as long as the process is active
        try:
            while self._process.poll() is None:
                result_stdout = read_stdout()
                result_stderr = read_stderr()

                if not result_stdout and not result_stderr:
                    sleep(self._refresh_time)

            # read remaining lines
            while read_stdout():
                pass

            while read_stderr():
                pass

        except (StopTask, AbortWorkflow) as exc:
            self._exc_obj = exc