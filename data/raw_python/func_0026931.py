def commands(self, event):
        """
        Lists all available commands.
        """
        commands = sorted(self.commands_dict().keys())
        return "Available commands: %s" % " ".join(commands)