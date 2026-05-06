def magic_generate_random(self, line):
        """
        Set the generate random flag, unset registers and memory will return a random value.

        Usage:
        Call the magic by itself or with `true` to have registers and memory return a random value
        if they are unset and read from, much like how real hardware would work.
        Defaults to False, or to not generate random values

        `%generate_random`
        or
        `%generate_random true`
        or
        `%generate_random false`
        """
        line = line.strip().lower()
        if not line or line == 'true':
            self.interpreter.generate_random = True
        elif line == 'false':
            self.interpreter.generate_random = False
        else:
            stream_content = {'name': 'stderr', 'text': "unknwon value '{}'".format(line)}
            self.send_response(self.iopub_socket, 'stream', stream_content)
            return {'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': ValueError.__name__,
                    'evalue': "unknwon value '{}'".format(line),
                    'traceback': '???'}