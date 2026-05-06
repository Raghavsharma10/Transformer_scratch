def unpack(self, target_dir=None, password=None, pickle_fname=None):
        """
        High level function for unpacking a backup file into the given
        target directory (will be generated based on the filename if not given).

        Creates also a filename.pickle file containing the exact order of the included files
        (required for repacking).

        :param target_dir: the directory to extract the backup file into
                           (default: filename + _unpacked)
        :param password: optional password for decrypting the backup
                         (can also be set in the constructor)
        """

        if target_dir is None:
           target_dir = os.path.basename(self.fname) + '_unpacked'
        if pickle_fname is None:
            pickle_fname = os.path.basename(self.fname) + '.pickle'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        tar = self.read_data(password)
        members = tar.getmembers()

        # reopen stream (TarFile is not able to seek)
        tar = self.read_data(password)

        tar.extractall(path=target_dir, members=members)

        with open(pickle_fname, 'wb') as fp:
            pickle.dump(members, fp)