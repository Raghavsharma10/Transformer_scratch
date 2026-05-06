def _touch_dir(self, path):
        """
        A helper function to create a directory if it doesn't exist.

        path: A string containing a full path to the directory to be created.
        """
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise