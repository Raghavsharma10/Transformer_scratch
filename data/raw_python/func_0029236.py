def get_command(domain_name, command_name):
    """Returns a closure function that dispatches message to the WebSocket."""
    def send_command(self, **kwargs):
        return self.ws.send_message(
            '{0}.{1}'.format(domain_name, command_name),
            kwargs
        )

    return send_command