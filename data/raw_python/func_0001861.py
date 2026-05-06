def render_audio(self):
        """
        Synthesize audio from the episode's text.
        """
        segment = text_to_speech(self._text, self.synthesizer, self.synth_args, self.sentence_break)

        milli = len(segment)
        seconds = '{0:.1f}'.format(float(milli) / 1000 % 60).zfill(2)
        minutes = '{0:.0f}'.format((milli / (1000 * 60)) % 60).zfill(2)
        hours = '{0:.0f}'.format((milli / (1000 * 60 * 60)) % 24).zfill(2)
        self.duration = hours + ':' + minutes + ':' + seconds

        segment.export(self.link, format='mp3')
        self.length = os.path.getsize(self.link)