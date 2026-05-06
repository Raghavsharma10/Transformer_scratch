def retrieve(self, filename):
        """
            Run the FTP RETR command, and download the file

        :param filename: Name of the file to download
        """
        logger.debug('Sending FTP retr command. Filename: {}'.format(filename))
        self.client.retrbinary('RETR {}'.format(filename), self._save_file)