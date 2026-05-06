def cancel(self, consumer_tag):
        """Cancel a channel by consumer tag."""
        if not self.channel.connection:
            return
        self.channel.basic_cancel(consumer_tag)