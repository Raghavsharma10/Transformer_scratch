def messages_in_flight(self):
        """
        Returns True if there are messages waiting to be sent or that we're
        still waiting to see if errors occur for.
        """
        self.prune_sent()
        if not self.send_queue.empty() or len(self.sent) > 0:
            return True
        return False