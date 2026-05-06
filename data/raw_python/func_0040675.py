def stage_subset(self, *files_to_add: str):
        """
        Stages a subset of files

        :param files_to_add: files to stage
        :type files_to_add: str
        """
        LOGGER.info('staging files: %s', files_to_add)
        self.repo.git.add(*files_to_add, A=True)