def load(self, path):
        """Load costume from image file.

        Uses :attr:`Image.load`, but will set the Costume's name based on the
        image filename.

        """
        (folder, filename) = os.path.split(path)
        (name, extension) = os.path.splitext(filename)
        return Costume(name, Image.load(path))