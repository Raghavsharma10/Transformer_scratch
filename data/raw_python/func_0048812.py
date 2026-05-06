def commands(self):
        """Memoize and return the flattened list of commands"""
        if not getattr(self, "_commands", None):
            self._commands = []
            for command in self.orig_commands:
                if isinstance(command, Command):
                    command = [command]
                for instruction in command:
                    self._commands.append(instruction)
        return self._commands