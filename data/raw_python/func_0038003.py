def _subprocess_method(self, command):
        """Use the subprocess module to execute ipmitool commands
        and and set status
        """
        p = subprocess.Popen([self._ipmitool_path] + self.args + command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.error = p.communicate()
        self.status = p.returncode