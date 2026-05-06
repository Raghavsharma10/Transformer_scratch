def list(self):
        """
            Run the FTP LIST command, and update the state.
        """
        logger.debug('Sending FTP list command.')
        self.state['file_list'] = []
        self.state['dir_list'] = []
        self.client.retrlines('LIST', self._process_list)