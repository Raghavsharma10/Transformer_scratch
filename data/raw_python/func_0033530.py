def get_all_feedback(self):
        """
        Connects to the feedback service and returns any feedback that is sent
        as a list of FeedbackItem objects.

        Blocks the current greenlet until all feedback is returned.

        If a network error occurs before any feedback is received, it is
        propagated to the caller. Otherwise, it is ignored and the feedback
        that had arrived is returned.
        """
        if not self.fbaddress:
            raise Exception("Attempted to fetch feedback but no feedback_address supplied")

        fbconn = FeedbackConnection(self, self.fbaddress, self.certfile, self.keyfile)
        return fbconn.get_all()