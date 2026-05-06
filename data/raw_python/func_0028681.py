def _process_event(self, event):
        """ Extend event object with User and Channel objects """
        if event.get('user'):
            event.user = self.lookup_user(event.get('user'))

        if event.get('channel'):
            event.channel = self.lookup_channel(event.get('channel'))

        if self.user.id in event.mentions:
            event.mentions_me = True

        event.mentions = [ self.lookup_user(uid) for uid in event.mentions ]

        return event