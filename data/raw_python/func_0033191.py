def _get_base_command(self):
        """Returns the base command plus command-line options.

        Does not include input file, output file, and training set.
        """
        cd_command = ''.join(['cd ', str(self.WorkingDir), ';'])
        jvm_command = "java"
        jvm_arguments = self._commandline_join(
            [self.Parameters[k] for k in self._jvm_parameters])
        jar_arguments = '-jar "%s"' % self._get_jar_fp()
        rdp_arguments = self._commandline_join(
            [self.Parameters[k] for k in self._options])

        command_parts = [
            cd_command, jvm_command, jvm_arguments, jar_arguments,
            rdp_arguments, '-q']
        return self._commandline_join(command_parts).strip()