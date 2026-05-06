def call_cmd(self, cmd, cwd):
        """
        Calls a command with Popen.
        Writes stdout, stderr, and the command to separate files.

        :param cmd: A string or array of strings.
        :param tempdir:
        :return: The pid of the command.
        """
        with open(self.cmdfile, 'w') as f:
            f.write(str(cmd))
        stdout = open(self.outfile, 'w')
        stderr = open(self.errfile, 'w')
        logging.info('Calling: ' + ' '.join(cmd))
        process = subprocess.Popen(cmd,
                                   stdout=stdout,
                                   stderr=stderr,
                                   close_fds=True,
                                   cwd=cwd)
        stdout.close()
        stderr.close()

        return process.pid