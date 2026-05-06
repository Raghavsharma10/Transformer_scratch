def check(self):
        """
        Checks if the list of tracked terms has changed.
        Returns True if changed, otherwise False.
        """

        new_tracking_terms = self.update_tracking_terms()

        terms_changed = False

        # any deleted terms?
        if self._tracking_terms_set > new_tracking_terms:
            logging.debug("Some tracking terms removed")
            terms_changed = True

        # any added terms?
        elif self._tracking_terms_set < new_tracking_terms:
            logging.debug("Some tracking terms added")
            terms_changed = True

        # Go ahead and store for later
        self._tracking_terms_set = new_tracking_terms

        # If the terms changed, we need to restart the stream
        return terms_changed