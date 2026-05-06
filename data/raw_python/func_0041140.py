def validate_context(self):
        """
        Make sure there are no duplicate context objects
        or we might end up with switched data

        Converting the tuple to a set gets rid of the
        eventual duplicate objects, comparing the length
        of the original tuple and set tells us if we
        have duplicates in the tuple or not
        """
        if self.context and len(self.context) != len(set(self.context)):
            LOGGER.error('Cannot have duplicated context objects')
            raise Exception('Cannot have duplicated context objects.')