def cwd(self, newdir):
        """
            Send the FTP CWD command

        :param newdir: Directory to change to
        """
        logger.debug('Sending FTP cwd command. New Workding Directory: {}'.format(newdir))
        self.client.cwd(newdir)
        self.state['current_dir'] = self.client.pwd()