def part(self, target, message=None):
        """Part a channel."""
        if not self.server.in_channel(target):
            _log.warning("Ignoring request to part channel '%s' because we "
                         "are not in that channel.", target)
            return
            return False
        self.send("PART", target, *([message] if message else []))
        return True