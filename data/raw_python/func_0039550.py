def _genpath(self, filename, mhash):
        """Generate the path to a file in the cache.

        Does not check to see if the file exists. Just constructs the path
        where it should be.
        """
        mhash = mhash.hexdigest()
        return os.path.join(self.mh_cachedir, mhash[0:2], mhash[2:4],
                            mhash, filename)