def add(self, session):
        """ Add session to the container.

        @param session: Session object
        """
        self._items[session.session_id] = session

        if session.expiry is not None:
            self._queue.push(session)