def herald_message(self, herald_svc, message):
        """
        Handles a message received by Herald

        :param herald_svc: Herald service
        :param message: Received message
        """
        subject = message.subject
        if subject == SUBJECT_DISCOVERY_STEP_1:
            # Step 1: Register the remote peer and reply with our dump
            try:
                # Delayed registration
                notification = self._directory.register_delayed(
                    self.__load_dump(message))

                peer = notification.peer
                if peer is not None:
                    # Registration succeeded
                    self.__delayed_notifs[peer.uid] = notification

                    # Reply with our dump
                    herald_svc.reply(
                        message, self._directory.get_local_peer().dump(),
                        SUBJECT_DISCOVERY_STEP_2)
            except ValueError:
                self._logger.error("Error registering a discovered peer")

        elif subject == SUBJECT_DISCOVERY_STEP_2:
            # Step 2: Register the dump, notify local listeners, then let
            # the remote peer notify its listeners
            try:
                # Register the peer
                notification = self._directory.register_delayed(
                    self.__load_dump(message))

                if notification.peer is not None:
                    # Let the remote peer notify its listeners
                    herald_svc.reply(message, None, SUBJECT_DISCOVERY_STEP_3)

                    # Now we can notify listeners
                    notification.notify()
            except ValueError:
                self._logger.error("Error registering a peer using the "
                                   "description it sent")

        elif subject == SUBJECT_DISCOVERY_STEP_3:
            # Step 3: notify local listeners about the remote peer
            try:
                self.__delayed_notifs.pop(message.sender).notify()
            except KeyError:
                # Unknown peer
                pass
        else:
            # Unknown subject
            self._logger.warning("Unknown discovery step: %s", subject)