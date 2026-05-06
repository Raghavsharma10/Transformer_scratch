def gen_localdir(self, localdir):
        """
        Generate local directory where track will be saved.
        Create it if not exists.
        """
        directory = "{0}/{1}/".format(localdir, self.get("username"))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory