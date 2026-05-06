def handle_quit(self, connection, event):
        """
        Store quit time for a nickname when it quits.
        """
        nickname = self.get_nickname(event)
        self.quit[nickname] = datetime.now()
        del self.joined[nickname]