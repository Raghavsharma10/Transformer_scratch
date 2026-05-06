def get_ignored_tracks(self, localdir):
        """ Get ignored tracks list. """
        ignore_file = "%s/.ignore" % localdir
        list = []
        if os.path.exists(ignore_file):
            f = open(ignore_file)
            ignored = f.readlines()
            f.close()

            for i in ignored:
                list.append("%s/%s" % (localdir, i.rstrip()))

        return list