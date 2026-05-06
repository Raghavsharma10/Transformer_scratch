def download_all(self, dst_dir=None):
        """Download all available files.

        Arguments:
        dst_dir   -- Optional destination directory to write files to.  If not
                     specified, then files are downloaded current directory.

        Return:
        Dictionary of {file_name: file_size, ..}

        """
        saved = {}
        save_as = None
        for f in self.files():
            if dst_dir:
                save_as = os.path.join(dst_dir, f.split('/')[-1])
            name, bytes = self.download(f, save_as)
            saved[name] = bytes
        return saved