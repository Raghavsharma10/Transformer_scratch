def redeliver(self):
        """
        Re-deliver the answer to the consequence which previously handled it
        by raising an exception.

        This method is intended to be invoked after the code in question has
        been upgraded.  Since there are no buggy answer receivers in
        production, nothing calls it yet.
        """
        self.consequence.answerReceived(self.answerValue,
                                        self.messageValue,
                                        self.sender,
                                        self.target)
        self.deleteFromStore()