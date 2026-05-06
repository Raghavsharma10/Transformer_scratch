def read_wav(self, filename):
        """Read sample data for this sample from a WAV file.

        :param filename: the file from which to read
        """
        wave_input = None

        try:
            wave_input = wave.open(filename, 'r')
            wave_frames = bytearray(
                wave_input.readframes(wave_input.getnframes()))

            self.sample_data = [x >> 4 for x in wave_frames]

        finally:
            if wave_input is not None:
                wave_input.close()