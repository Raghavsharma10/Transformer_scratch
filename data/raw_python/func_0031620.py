def send_stop_audio(self):
        '''
        Stop an audio streaming session
        '''
        assert self._session_id != VoiceService.SESSION_ID_INVALID

        self._pebble.send_packet(AudioStream(session_id=self._session_id, data=StopTransfer()))