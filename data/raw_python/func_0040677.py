def commit(
            self,
            message: str,
            files_to_add: typing.Optional[typing.Union[typing.List[str], str]] = None,
            allow_empty: bool = False,
    ):
        """
        Commits changes to the repo

        :param message: first line of the message
        :type message: str
        :param files_to_add: files to commit
        :type files_to_add: optional list of str
        :param allow_empty: allow dummy commit
        :type allow_empty: bool
        """
        message = str(message)
        LOGGER.debug('message: %s', message)

        files_to_add = self._sanitize_files_to_add(files_to_add)
        LOGGER.debug('files to add: %s', files_to_add)

        if not message:
            LOGGER.error('empty commit message')
            sys.exit(-1)

        if os.getenv('APPVEYOR'):
            LOGGER.info('committing on AV, adding skip_ci tag')
            message = self.add_skip_ci_to_commit_msg(message)

        if files_to_add is None:
            self.stage_all()
        else:
            self.reset_index()
            self.stage_subset(*files_to_add)

        if self.index_is_empty() and not allow_empty:
            LOGGER.error('empty commit')
            sys.exit(-1)

        self.repo.index.commit(message=message)