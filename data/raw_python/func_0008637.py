def save(self, path):
        """Save the sound to a wave file at the given path.

        Uses :attr:`Waveform.save`, but if the path ends in a folder instead of
        a file, the filename is based on the project's :attr:`name`.

        :returns: Path to the saved file.

        """
        (folder, filename) = os.path.split(path)
        if not filename:
            filename = _clean_filename(self.name)
            path = os.path.join(folder, filename)
        return self.waveform.save(path)