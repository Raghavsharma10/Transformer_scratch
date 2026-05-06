def find_similar(self, *args, **kwargs):
        """
        Find similar articles.

        With autosession off, use the index state *before* current session started,
        so that changes made in the session will not be visible here. With autosession
        on, close the current session first (so that session changes *are* committed
        and visible).
        """
        if self.session is not None and self.autosession:
            # with autosession on, commit the pending transaction first
            self.commit()
        return self.stable.find_similar(*args, **kwargs)