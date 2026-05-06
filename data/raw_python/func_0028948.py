def magic_help(self, line):
        """
        Print out the help for magics

        Usage:
        Call help with no arguments to list all magics,
        or call it with a magic to print out it's help info.

        `%help`
        or
        `%help run
        """
        line = line.strip()
        if not line:
            for magic in self.magics:
                stream_content = {'name': 'stdout', 'text': "%{}\n".format(magic)}
                self.send_response(self.iopub_socket, 'stream', stream_content)
        elif line in self.magics:
            # its a magic
            stream_content = {'name': 'stdout', 'text': "{}\n{}".format(line, self.magics[line].__doc__)}
            self.send_response(self.iopub_socket, 'stream', stream_content)
        elif line in self.interpreter.ops:
            # it's an instruction
            stream_content = {'name': 'stdout', 'text': "{}\n{}".format(line, self.interpreter.ops[line].__doc__)}
            self.send_response(self.iopub_socket, 'stream', stream_content)
        else:
            stream_content = {'name': 'stderr', 'text': "'{}' not a known magic or instruction".format(line)}
            self.send_response(self.iopub_socket, 'stream', stream_content)