def remove_dir(self, destination_path):
        """
        Remove folder. Based on https://gist.github.com/artlogic/2632647.
        """

        wd = self.conn.pwd()

        try:
            names = self.conn.nlst(destination_path)
        except ftplib.all_errors as e:
            # some FTP servers complain when you try and list non-existent paths
            logger.debug('FtpRmTree: Could not remove {0}: {1}'.format(
                destination_path, e))
            return

        for name in names:
            if os.path.split(name)[1] in ('.', '..'):
                continue

            try:
                self.conn.cwd(name)  # if we can cwd to it, it's a folder
                self.conn.cwd(wd)  # don't try a nuke a folder we're in
                self.remove_dir(name)
            except ftplib.all_errors:
                self.conn.delete(name)

        try:
            self.conn.rmd(destination_path)
        except ftplib.all_errors as e:
            logger.debug('remove_dir: Could not remove {0}: {1}'.format(
                destination_path, e))