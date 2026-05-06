def on_pubmsg(self, connection, event):
        """
        Log any public messages, and also handle the command event.
        """
        for message in event.arguments():
            self.log(event, message)
            command_args = filter(None, message.split())
            command_name = command_args.pop(0)
            for handler in self.events["command"]:
                if handler.event.args["command"] == command_name:
                    self.handle_command_event(event, handler, command_args)