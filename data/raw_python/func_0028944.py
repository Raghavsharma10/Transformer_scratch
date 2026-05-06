def magic_postpone_execution(self, line):
        """
        Postpone execution of instructions until explicitly run

        Usage:
        Call this magic with `true` or nothing to postpone execution,
        or call with `false` to execute each instruction when evaluated.
        This defaults to True.

        Note that each cell is executed only executed after all lines in
        the cell have been evaluated properly.

        `%postpone_execution`
        or
        `%postpone_execution true`
        or
        `%postpone_execution false`
        """
        line = line.strip().lower()
        if not line or line == 'true':
            self.interpreter.postpone_execution = True
        elif line == 'false':
            self.interpreter.postpone_execution = False
        else:
            stream_content = {'name': 'stderr', 'text': "unknwon value '{}'".format(line)}
            self.send_response(self.iopub_socket, 'stream', stream_content)
            return {'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': ValueError.__name__,
                    'evalue': "unknwon value '{}'".format(line),
                    'traceback': '???'}