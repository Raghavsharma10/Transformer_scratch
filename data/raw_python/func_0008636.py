def load(self, path):
        """Load sound from wave file.

        Uses :attr:`Waveform.load`, but will set the Waveform's name based on
        the sound filename.

        """
        (folder, filename) = os.path.split(path)
        (name, extension) = os.path.splitext(filename)
        return Sound(name, Waveform.load(path))