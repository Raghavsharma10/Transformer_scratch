def formatted_command(self):
        """
        If we have ``bash``, then the command is ``/bin/bash -c <bash>``, whereas
        if the ``command`` is set, then we just return that.
        """
        bash = self.bash
        if bash not in (None, "", NotSpecified) and callable(bash):
            bash = bash()
        if bash not in (None, "", NotSpecified):
            return "{0} -c {1}".format(self.resolved_shell, shlex_quote(bash))

        command = self.command
        if command not in (None, "", NotSpecified) and callable(command):
            command = command()
        if command not in (None, "", NotSpecified):
            return command

        return None