def get_available_name(self, name, max_length=None):
        """Return relative path to name placed in random directory"""
        tempdir = tempfile.mkdtemp(dir=self.base_location)
        name = os.path.join(
            os.path.basename(tempdir),
            os.path.basename(name),
        )
        method = super(TempFileSystemStorage, self).get_available_name
        return method(name, max_length=max_length)