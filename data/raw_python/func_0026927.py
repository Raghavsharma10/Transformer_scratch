def handle_join(self, connection, event):
        """
        Store join time for a nickname when it joins.
        """
        nickname = self.get_nickname(event)
        self.joined[nickname] = datetime.now()