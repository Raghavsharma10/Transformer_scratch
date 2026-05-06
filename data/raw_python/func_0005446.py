def run(self, command, block=True, cwd=None, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
        """
        Create an instance of :class:`~ShellCommand` and run it

        Args:
            command (str): :class:`~ShellCommand`
            block (bool): See :class:`~ShellCommand`
            cwd (str): Override the runner cwd. Useb by the :class:`~ShellCommand` instance
        """
        if cwd is None:
            cwd = self.cwd

        return ShellCommand(command=command, logger=self.logger, block=block, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr).run()