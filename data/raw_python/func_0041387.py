def gather(self, cmd):
        """
        Runs a command and returns rc,stdout,stderr as a tuple.

        If called while the `Dir` context manager is in effect, guarantees that the
        process is executed in that directory, even if it is no longer the current
        directory of the process (i.e. it is thread-safe).

        :param cmd: The command and arguments to execute
        :return: (rc,stdout,stderr)
        """

        if not isinstance(cmd, list):
            cmd_list = shlex.split(cmd)
        else:
            cmd_list = cmd

        cwd = pushd.Dir.getcwd()
        cmd_info = '[cwd={}]: {}'.format(cwd, cmd_list)

        self.logger.debug("Executing:gather {}".format(cmd_info))
        proc = subprocess.Popen(
            cmd_list, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        rc = proc.returncode
        self.logger.debug(
            "Process {}: exited with: {}\nstdout>>{}<<\nstderr>>{}<<\n".
            format(cmd_info, rc, out, err))
        return rc, out, err