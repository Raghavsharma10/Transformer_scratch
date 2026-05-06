def create_from_file(cls, filename):
        """Return an Estimator object given the path of the file, relative to the MEDIA_ROOT"""
        obj = cls()
        obj.object_file = filename
        obj.load()
        return obj