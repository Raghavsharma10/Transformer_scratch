def handle_command_event(self, event, command, args):
        """
        Command handler - treats each word in the message
        that triggered the command as an argument to the command,
        and does some validation to ensure that the number of
        arguments match.
        """
        argspec = getargspec(command)
        num_all_args = len(argspec.args) - 2  # Ignore self/event args
        num_pos_args = num_all_args - len(argspec.defaults or [])
        if num_pos_args <= len(args) <= num_all_args:
            response = command(self, event, *args)
        elif num_all_args == num_pos_args:
            s = "s are" if num_all_args != 1 else " is"
            response = "%s arg%s required" % (num_all_args, s)
        else:
            bits = (num_pos_args, num_all_args)
            response = "between %s and %s args are required" % bits
        response = "%s: %s" % (self.get_nickname(event), response)
        self.message_channel(response)