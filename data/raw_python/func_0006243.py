def get_matching_session(self, session, db_session, timediff=5):
        """
        Tries to match a session with it's counterpart. For bait session it will try to match it with honeypot sessions
        and the other way around.

        :param session: session object which will be used as base for query.
        :param timediff: +/- allowed time difference between a session and a potential matching session.
        """
        db_session = db_session
        min_datetime = session.timestamp - timedelta(seconds=timediff)
        max_datetime = session.timestamp + timedelta(seconds=timediff)
        # default return value
        match = None
        classification = db_session.query(Classification).filter(
            Classification.type == 'pending').one()
        # get all sessions that match basic properties.
        sessions = db_session.query(Session).options(joinedload(Session.authentication)) \
            .filter(Session.protocol == session.protocol) \
            .filter(Session.honeypot == session.honeypot) \
            .filter(Session.timestamp >= min_datetime) \
            .filter(Session.timestamp <= max_datetime) \
            .filter(Session.id != session.id) \
            .filter(Session.classification == classification)

        # identify the correct session by comparing authentication.
        # this could properly also be done using some fancy ORM/SQL construct.
        for potential_match in sessions:
            if potential_match.discriminator == session.discriminator:
                continue
            assert potential_match.id != session.id
            for honey_auth in session.authentication:
                for session_auth in potential_match.authentication:
                    if session_auth.username == honey_auth.username and \
                            session_auth.password == honey_auth.password and \
                            session_auth.successful == honey_auth.successful:
                        assert potential_match.id != session.id
                        match = potential_match
                        break

        return match