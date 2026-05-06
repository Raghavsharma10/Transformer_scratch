def help(self, event, command_name=None):
        """
        Shows the help message for the bot. Takes an optional command name
        which when given, will show help for that command.
        """
        if command_name is None:
            return ("Type !commands for a list of all commands. Type "
                    "!help [command] to see help for a specific command.")
        try:
            command = self.commands_dict()[command_name]
        except KeyError:
            return "%s is not a command" % command_name

        argspec = getargspec(command)
        args = argspec.args[2:]
        defaults = argspec.defaults or []
        for i in range(-1, -len(defaults) - 1, -1):
            args[i] = "%s [default: %s]" % (args[i], defaults[i])
        args = ", ".join(args)
        help = getdoc(command).replace("\n", " ")
        return "help for %s: (args: %s) %s" % (command_name, args, help)