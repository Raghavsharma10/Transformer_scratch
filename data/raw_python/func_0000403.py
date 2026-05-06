def process_action(self):
        """
        Process the action and update the related object, returns a boolean if a change is made.
        """
        if self.publish_version == self.UNPUBLISH_CHOICE:
            actioned = self._unpublish()
        else:
            actioned = self._publish()

        # Only log if an action was actually taken
        if actioned:
            self._log_action()

        return actioned