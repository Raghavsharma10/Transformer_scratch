def _get_command(self):
        """Return command with options and targets, ready for execution."""
        targets = ' '.join(self.targets)
        cmd_str = self._linter.command_with_options + ' ' + targets
        cmd_shlex = shlex.split(cmd_str)
        return list(cmd_shlex)