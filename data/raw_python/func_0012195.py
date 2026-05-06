def run(self):
        '''Execute the expression and return a Result, which includes the exit
        status and any captured output. Raise an exception if the status is
        non-zero.'''
        with spawn_output_reader() as (stdout_capture, stdout_thread):
            with spawn_output_reader() as (stderr_capture, stderr_thread):
                context = starter_iocontext(stdout_capture, stderr_capture)
                status = self._exec(context)
        stdout_bytes = stdout_thread.join()
        stderr_bytes = stderr_thread.join()
        result = Result(status.code, stdout_bytes, stderr_bytes)
        if is_checked_error(status):
            raise StatusError(result, self)
        return result