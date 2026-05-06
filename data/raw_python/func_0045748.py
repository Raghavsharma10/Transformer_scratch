def transmit_agnocomplete_context(self):
        """
        Assign the user context to the agnocomplete class, if any.
        """
        # Only if the field has this attribute set.
        if hasattr(self, AGNOCOMPLETE_USER_ATTRIBUTE):
            user = self.get_agnocomplete_context()
            if user:
                self.agnocomplete.user = user
            return user