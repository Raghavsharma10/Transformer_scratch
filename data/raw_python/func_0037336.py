def play(state):
        """ Play sound for a given state.

        :param state: a State value.
        """
        filename = None
        if state == SoundService.State.welcome:
            filename = "pad_glow_welcome1.wav"
        elif state == SoundService.State.goodbye:
            filename = "pad_glow_power_off.wav"
        elif state == SoundService.State.hotword_detected:
            filename = "pad_soft_on.wav"
        elif state == SoundService.State.asr_text_captured:
            filename = "pad_soft_off.wav"
        elif state == SoundService.State.error:
            filename = "music_marimba_error_chord_2x.wav"

        if filename is not None:
            AudioPlayer.play_async("{}/{}".format(ABS_SOUND_DIR, filename))