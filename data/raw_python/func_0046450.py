def join(self, target, key=None):
        """Attempt to join a channel.

        The optional second argument is the channel key, if needed.
        """
        chantypes = self.server.features.get("CHANTYPES", "#")
        if not target or target[0] not in chantypes:
            # Among other things, this prevents accidentally sending the
            # "JOIN 0" command which actually removes you from all channels
            _log.warning("Refusing to join channel that does not start "
                         "with one of '%s': %s", chantypes, target)
            return False

        if self.server.in_channel(target):
            _log.warning("Ignoring request to join channel '%s' because we "
                         "are already in that channel.", target)
            return False

        _log.info("Joining channel %s ...", target)
        self.send("JOIN", target, *([key] if key else []))
        return True