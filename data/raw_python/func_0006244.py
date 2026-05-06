def _classify_malicious_sessions(self):
        """
        Will classify all unclassified sessions as malicious activity.

        :param delay_seconds: no sessions newer than (now - delay_seconds) will be processed.
        """
        min_datetime = datetime.utcnow() - timedelta(seconds=self.delay_seconds)

        db_session = database_setup.get_session()

        # find and process bait sessions that did not get classified during
        # persistence.
        bait_sessions = db_session.query(BaitSession).options(joinedload(BaitSession.authentication)) \
            .filter(BaitSession.classification_id == 'pending') \
            .filter(BaitSession.did_complete == True) \
            .filter(BaitSession.received < min_datetime).all()

        for bait_session in bait_sessions:
            logger.debug(
                'Classifying bait session with id {0} as MITM'.format(bait_session.id))
            bait_session.classification = db_session.query(
                Classification).filter(Classification.type == 'mitm').one()
            db_session.commit()

        # find and process honeypot sessions that did not get classified during
        # persistence.
        sessions = db_session.query(Session, Drone.name).filter(Session.discriminator == None) \
            .filter(Session.timestamp <= min_datetime) \
            .filter(Session.classification_id == 'pending') \
            .all()

        for entry in sessions:
            # Check if the attack used credentials leaked by beeswarm drones
            session = entry[0]
            bait_match = None
            for a in session.authentication:
                bait_match = db_session.query(BaitSession) \
                    .filter(BaitSession.authentication.any(username=a.username, password=a.password)).first()
                if bait_match:
                    break

            if bait_match:
                logger.debug('Classifying session with id {0} as attack which involved the reuse '
                             'of previously transmitted credentials.'.format(session.id))
                session.classification = db_session.query(Classification).filter(
                    Classification.type == 'credentials_reuse').one()
            elif len(session.authentication) == 0:
                logger.debug(
                    'Classifying session with id {0} as probe.'.format(session.id))
                session.classification = db_session.query(
                    Classification).filter(Classification.type == 'probe').one()
            else:
                # we have never transmitted this username/password combo
                logger.debug(
                    'Classifying session with id {0} as bruteforce attempt.'.format(session.id))
                session.classification = db_session.query(Classification).filter(
                    Classification.type == 'bruteforce').one()
            db_session.commit()
            session.name = entry[1]
            self.processedSessionsPublisher.send(
                '{0} {1}'.format(Messages.SESSION.value, json.dumps(session.to_dict())))