def str_uptime(self):
        """uptime in human readable format."""
        mins, secs = divmod(self.uptime, 60)
        hours, mins = divmod(mins, 60)
        return '%02d:%02d:%02d' % (hours, mins, secs)