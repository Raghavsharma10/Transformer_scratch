def send_dictation_result(self, result, sentences=None, app_uuid=None):
        '''
        Send the result of a dictation session

        :param result: Result of the session
        :type result: DictationResult
        :param sentences: list of sentences, each of which is a list of words and punctuation
        :param app_uuid: UUID of app that initiated the session
        :type app_uuid: uuid.UUID
        '''

        assert self._session_id != VoiceService.SESSION_ID_INVALID
        assert isinstance(result, TranscriptionResult)

        transcription = None
        if result == TranscriptionResult.Success:
            if len(sentences) > 0:
                s_list = []
                for s in sentences:
                    words = [Word(confidence=100, data=w) for w in s]
                    s_list.append(Sentence(words=words))
                transcription = Transcription(transcription=SentenceList(sentences=s_list))

        flags = 0
        if app_uuid is not None:
            assert isinstance(app_uuid, uuid.UUID)
            flags |= Flags.AppInitiated

        attributes = []
        if app_uuid is not None:
            assert isinstance(app_uuid, uuid.UUID)
            attributes.append(Attribute(id=AttributeType.AppUuid, data=AppUuid(uuid=app_uuid)))

        if transcription is not None:
            attributes.append(Attribute(id=AttributeType.Transcription, data=transcription))

        logger.debug("Sending dictation result (result={}".format(result) +
                     ", app={})".format(app_uuid) if app_uuid is not None else ")")
        self._pebble.send_packet(VoiceControlResult(flags=flags, data=DictationResult(
            session_id=self._session_id, result=result, attributes=AttributeList(dictionary=attributes))))
        self._session_id = VoiceService.SESSION_ID_INVALID