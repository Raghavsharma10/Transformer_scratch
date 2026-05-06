def play(self, filename, translate=False):  # pragma: no cover
        '''
        Plays the sounds.

        :filename: The input file name
        :translate: If True, it runs it through audioread which will translate from common compression formats to raw WAV.
        '''
        # FIXME: Use platform-independent and async audio-output here
        # PyAudio looks most promising, too bad about:
        #  --allow-external PyAudio --allow-unverified PyAudio
        if translate:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                fname = f.name
            with audioread.audio_open(filename) as f:
                with contextlib.closing(wave.open(fname, 'w')) as of:
                    of.setnchannels(f.channels)
                    of.setframerate(f.samplerate)
                    of.setsampwidth(2)
                    for buf in f:
                        of.writeframes(buf)
            filename = fname

        if winsound:
            winsound.PlaySound(str(filename), winsound.SND_FILENAME)
        else:
            cmd = ['aplay', str(filename)]
            self._logger.debug('Executing %s', ' '.join([pipes.quote(arg) for arg in cmd]))
            subprocess.call(cmd)

        if translate:
            os.remove(fname)