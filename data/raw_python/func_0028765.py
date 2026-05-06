def setTimeout(self, time):
        """Set global timeout value, in seconds, for all DDE calls"""
        self.conversation.SetDDETimeout(round(time))
        return self.conversation.GetDDETimeout()