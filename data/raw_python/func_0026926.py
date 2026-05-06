def handle_joined(self, connection, event):
        """
        Store join times for current nicknames when we first join.
        """
        nicknames = [s.lstrip("@+") for s in event.arguments()[-1].split()]
        for nickname in nicknames:
            self.joined[nickname] = datetime.now()