def seen(self, event, nickname):
        """
        Shows the amount of time since the given nickname was last
        seen in the channel.
        """
        try:
            self.joined[nickname]
        except KeyError:
            pass
        else:
            if nickname == self.get_nickname(event):
                prefix = "you are"
            else:
                prefix = "%s is" % nickname
            return "%s here right now" % prefix
        try:
            seen = self.timesince(self.quit[nickname])
        except KeyError:
            return "%s has never been seen" % nickname
        else:
            return "%s was last seen %s ago" % (nickname,  seen)