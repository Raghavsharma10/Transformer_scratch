def load(cls, path):
        """Load Waveform from file."""
        assert os.path.exists(path), "No such file: %r" % path

        (folder, filename) = os.path.split(path)
        (name, extension) = os.path.splitext(filename)

        wave = Waveform(None)
        wave._path = path
        return wave