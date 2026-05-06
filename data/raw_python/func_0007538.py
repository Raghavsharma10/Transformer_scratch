def destroy(self):
        """Destroys a session completely, by deleting all keys and removing it
        from the internal store immediately.

        This allows removing a session for security reasons, e.g. a login
        stored in a session will cease to exist if the session is destroyed.
        """
        for k in list(self.keys()):
            del self[k]

        if getattr(self, 'sid_s', None):
            current_app.kvsession_store.delete(self.sid_s)
            self.sid_s = None

        self.modified = False
        self.new = False