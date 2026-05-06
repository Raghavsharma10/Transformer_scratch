def _get_base_command(self):
        """ Returns the full command string

            input_arg: the argument to the command which represents the input
                to the program, this will be a string, either
                representing input or a filename to get input from
         tI"""
        command_parts = []
        # Append a change directory to the beginning of the command to change
        # to self.WorkingDir before running the command
        # WorkingDir should be in quotes -- filenames might contain spaces
        cd_command = ''.join(['cd ', str(self.WorkingDir), ';'])
        if self._command is None:
            raise ApplicationError('_command has not been set.')
        command = self._command
        parameters = self.Parameters

        command_parts.append(cd_command)
        command_parts.append(command)
        command_parts.append(self._command_delimiter.join(filter(
            None, (map(str, parameters.values())))))

        return self._command_delimiter.join(command_parts).strip()