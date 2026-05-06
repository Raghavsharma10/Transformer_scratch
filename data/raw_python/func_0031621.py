def send_session_setup_result(self, result, app_uuid=None):
        '''
        Send the result of setting up a dictation session requested by the watch

        :param result:  result of setting up the session
        :type result: .SetupResult
        :param app_uuid: UUID of app that initiated the session
        :type app_uuid: uuid.UUID
        '''
        assert self._session_id != VoiceService.SESSION_ID_INVALID
        assert isinstance(result, SetupResult)

        flags = 0
        if app_uuid is not None:
            assert isinstance(app_uuid, uuid.UUID)
            flags |= Flags.AppInitiated

        logger.debug("Sending session setup result (result={}".format(result) +
                     ", app={})".format(app_uuid) if app_uuid is not None else ")")

        self._pebble.send_packet(VoiceControlResult(flags=flags, data=SessionSetupResult(
                session_type=SessionType.Dictation, result=result)))

        if result != SetupResult.Success:
            self._session_id = VoiceService.SESSION_ID_INVALID