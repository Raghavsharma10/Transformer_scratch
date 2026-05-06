def run(self):
        """
        Run the shell command

        Returns:
            ShellCommand: return this ShellCommand instance for chaining
        """
        if not self.block:
            self.output = []
            self.error = []
            self.thread = threading.Thread(target=self.run_non_blocking)
            self.thread.start()
        else:
            self.__create_process()
            self.process.wait()
            if self._stdout is not None:
                self.output = self.process.stdout.read().decode("utf-8")
            if self._stderr is not None:
                self.error = self.process.stderr.read().decode("utf-8")
            self.return_code = self.process.returncode

        return self