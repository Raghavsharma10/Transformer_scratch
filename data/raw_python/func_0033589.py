def _get_base_command(self):
        """ Returns the full command string

        Overridden here because there are positional arguments (specifically
        the input and output files).
        """
        command_parts = []
        # Append a change directory to the beginning of the command to change
        # to self.WorkingDir before running the command
        # WorkingDir should be in quotes -- filenames might contain spaces
        cd_command = ''.join(['cd ', str(self.WorkingDir), ';'])
        if self._command is None:
            raise ApplicationError('_command has not been set.')
        command = self._command
        # also make sure there's a subcommand!
        if self._subcommand is None:
            raise ApplicationError('_subcommand has not been set.')
        subcommand = self._subcommand
        # sorting makes testing easier, since the options will be written out
        # in alphabetical order. Could of course use option parsing scripts
        # in cogent for this, but this works as well.
        parameters = sorted([str(x) for x in self.Parameters.values()
                            if str(x)])
        synonyms = self._synonyms

        command_parts.append(cd_command)
        command_parts.append(command)
        # add in subcommand
        command_parts.append(subcommand)
        command_parts += parameters
        # add in the positional arguments in the correct order
        for k in self._input_order:
            # this check is necessary to account for optional positional
            # arguments, such as the mate file for bwa bwasw
            # Note that the input handler will ensure that all required
            # parameters have valid values
            if k in self._input:
                command_parts.append(self._input[k])

        return self._command_delimiter.join(command_parts).strip()