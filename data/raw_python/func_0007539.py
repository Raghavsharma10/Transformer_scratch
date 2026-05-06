def regenerate(self):
        """Generate a new session id for this session.

        To avoid vulnerabilities through `session fixation attacks
        <http://en.wikipedia.org/wiki/Session_fixation>`_, this function can be
        called after an action like a login has taken place. The session will
        be copied over to a new session id and the old one removed.
        """
        self.modified = True

        if getattr(self, 'sid_s', None):
            # delete old session
            current_app.kvsession_store.delete(self.sid_s)

            # remove sid_s, set modified
            self.sid_s = None
            self.modified = True