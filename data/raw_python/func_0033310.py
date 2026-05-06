def _get_base_command(self):
        """Gets the command that will be run when the app controller is
        called.
        """
        command_parts = []
        cd_command = ''.join(['cd ', str(self.WorkingDir), ';'])
        if self._command is None:
            raise ApplicationError('_command has not been set.')
        command = self._command
        parameters = sorted([str(x) for x in self.Parameters.values()
                            if str(x)])

        synonyms = self._synonyms

        command_parts.append(cd_command)
        command_parts.append(command)
        command_parts.append(self._database)  # Positional argument
        command_parts.append(self._query)  # Positional argument
        command_parts += parameters
        if self._output:
            command_parts.append(self._output.Path)  # Positional

        return (
            self._command_delimiter.join(filter(None, command_parts)).strip()
        )