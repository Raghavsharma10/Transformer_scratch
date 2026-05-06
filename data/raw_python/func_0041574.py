def push(self):
        """
        Push changes back to data repo.
        Will of course fail if user does not have write access.
        """
        self.logger.info('Pushing config...')
        with Dir(self.data_path):
            self.cmd.check_assert('git push')