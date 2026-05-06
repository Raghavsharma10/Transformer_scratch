def execute(self, command):
        '''
        Execute a subprocess yielding output lines
        '''
        process = Popen(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        while True:
            if process.poll() is not None:
                self.returncode = process.returncode  # pylint: disable=W0201
                break
            yield process.stdout.readline()