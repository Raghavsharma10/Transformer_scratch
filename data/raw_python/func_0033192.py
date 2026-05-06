def _commandline_join(self, tokens):
        """Formats a list of tokens as a shell command

        This seems to be a repeated pattern; may be useful in
        superclass.
        """
        commands = filter(None, map(str, tokens))
        return self._command_delimiter.join(commands).strip()