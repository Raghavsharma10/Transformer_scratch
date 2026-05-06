def attach_file(self, path, mimetype=None):
        """Attache a file from the filesystem."""
        filename = os.path.basename(path)
        content = open(path, "rb").read()
        self.attach(filename, content, mimetype)