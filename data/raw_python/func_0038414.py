def track_exists(self, localdir):
        """ Check if track exists in local directory. """
        path = glob.glob(self.gen_localdir(localdir) +
                         self.gen_filename() + "*")
        if len(path) > 0 and os.path.getsize(path[0]) > 0:
            return True
        return False