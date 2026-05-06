def _handle(self, nick, target, message, **kwargs):
        """ client callback entrance """
        for regex, (func, pattern) in self.routes.items():
            match = regex.match(message)
            if match:
                self.client.loop.create_task(
                    func(nick, target, message, match, **kwargs))