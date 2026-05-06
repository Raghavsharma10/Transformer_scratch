def magic_run(self, line):
        """
        Run the current program

        Usage:
        Call with a numbe rto run that many steps,
        or call with no arguments to run to the end of the current program

        `%run`
        or
        `%run 1`
        """
        i = float('inf')
        if line.strip():
            i = int(line)

        try:
            with warnings.catch_warnings(record=True) as w:
                self.interpreter.run(i)
                for warning_message in w:
                    # TODO should this be stdout or stderr
                    stream_content = {'name': 'stdout', 'text': 'Warning: ' + str(warning_message.message) + '\n'}
                    self.send_response(self.iopub_socket, 'stream', stream_content)
        except iarm.exceptions.EndOfProgram as e:
            f_name = self.interpreter.program[self.interpreter.register['PC'] - 1].__name__
            f_name = f_name[:f_name.find('_')]
            message = "Error in {}: ".format(f_name)
            stream_content = {'name': 'stdout', 'text': message + str(e) + '\n'}
            self.send_response(self.iopub_socket, 'stream', stream_content)
        except Exception as e:
            for err in e.args:
                stream_content = {'name': 'stderr', 'text': str(err)}
                self.send_response(self.iopub_socket, 'stream', stream_content)
            return {'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': '???'}