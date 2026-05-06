def _get_base_command(self):
        """Returns the base command plus command-line options.

        Handles everything up to and including the classpath.  The
        positional training parameters are added by the
        _input_handler_decorator method.
        """
        cd_command = ''.join(['cd ', str(self.WorkingDir), ';'])
        jvm_command = "java"
        jvm_args = self._commandline_join(
            [self.Parameters[k] for k in self._jvm_parameters])
        cp_args = '-cp "%s" %s' % (self._get_jar_fp(), self.TrainingClass)

        command_parts = [cd_command, jvm_command, jvm_args, cp_args]
        return self._commandline_join(command_parts).strip()