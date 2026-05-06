def load(cls, path):
        """Load image from file."""
        assert os.path.exists(path), "No such file: %r" % path

        (folder, filename) = os.path.split(path)
        (name, extension) = os.path.splitext(filename)

        image = Image(None)
        image._path = path
        image._format = Image.image_format(extension)

        return image