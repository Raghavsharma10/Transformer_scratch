def trigger(self, event_name, data):
        """Trigger an event on this channel.  Only available for private or
        presence channels

        :param event_name: The name of the event.  Must begin with 'client-''
        :type event_name: str

        :param data: The data to send with the event.
        """
        if self.connection:
            if event_name.startswith("client-"):
                if self.name.startswith("private-") or self.name.startswith("presence-"):
                    self.connection.send_event(event_name, data, channel_name=self.name)