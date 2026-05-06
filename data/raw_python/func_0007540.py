def cleanup_sessions(self, app=None):
        """Removes all expired session from the store.

        Periodically, this function can be called to remove sessions from
        the backend store that have expired, as they are not removed
        automatically unless the backend supports time-to-live and has been
        configured appropriately (see :class:`~simplekv.TimeToLiveMixin`).

        This function retrieves all session keys, checks they are older than
        :attr:`flask.Flask.permanent_session_lifetime` and if so, removes them.

        Note that no distinction is made between non-permanent and permanent
        sessions.

        :param app: The app whose sessions should be cleaned up. If ``None``,
                    uses :py:data:`~flask.current_app`."""

        if not app:
            app = current_app
        for key in app.kvsession_store.keys():
            m = self.key_regex.match(key)
            now = datetime.utcnow()
            if m:
                # read id
                sid = SessionID.unserialize(key)

                # remove if expired
                if sid.has_expired(app.permanent_session_lifetime, now):
                    app.kvsession_store.delete(key)