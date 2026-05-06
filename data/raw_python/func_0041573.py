def commit(self, msg):
        """
        Commit outstanding data changes
        """
        self.logger.info('Commit config: {}'.format(msg))
        with Dir(self.data_path):
            self.cmd.check_assert('git add .')
            self.cmd.check_assert('git commit --allow-empty -m "{}"'.format(msg))