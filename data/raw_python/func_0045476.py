def remove_token(self, act):
        """
        Remove a token from the proactor.
        If removal succeeds (the token is in the proactor) return True.
        """
        if act in self.tokens:
            self.unregister_fd(act)
            del self.tokens[act]
            return True
        else:
            import warnings
            warnings.warn("%s isn't a registered token." % act)