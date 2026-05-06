def _execute(self, command, stdin=None, stdout=subprocess.PIPE):
        """Executes the specified command relative to the repository root.
        Returns a tuple containing the return code and the process output.
        """
        process = subprocess.Popen(command, shell=True, cwd=self.root_path, stdin=stdin, stdout=stdout)
        return (process.wait(), None if stdout is not subprocess.PIPE else process.communicate()[0].decode('utf-8'))