def uptime(self, event, nickname=None):
        """
        Shows the amount of time since the given nickname has been
        in the channel. If no nickname is given, I'll use my own.
        """
        if nickname and nickname != self.nickname:
            try:
                uptime = self.timesince(self.joined[nickname])
            except KeyError:
                return "%s is not in the channel" % nickname
            else:
                if nickname == self.get_nickname(event):
                    prefix = "you have"
                else:
                    prefix = "%s has" % nickname
                return "%s been here for %s" % (prefix, uptime)
        uptime = self.timesince(self.joined[self.nickname])
        return "I've been here for %s" % uptime