def amend_commit(
            self,
            append_to_msg: typing.Optional[str] = None,
            new_message: typing.Optional[str] = None,
            files_to_add: typing.Optional[typing.Union[typing.List[str], str]] = None,
    ):
        """
        Amends last commit with either an entirely new commit message, or an edited version of the previous one

        Note: it is an error to provide both "append_to_msg" and "new_message"

        :param append_to_msg: message to append to previous commit message
        :type append_to_msg: str
        :param new_message: new commit message
        :type new_message: str
        :param files_to_add: optional list of files to add to this commit
        :type files_to_add: str or list of str
        """

        if new_message and append_to_msg:
            LOGGER.error('Cannot use "new_message" and "append_to_msg" together')
            sys.exit(-1)

        files_to_add = self._sanitize_files_to_add(files_to_add)

        message = self._sanitize_amend_commit_message(append_to_msg, new_message)

        if os.getenv('APPVEYOR'):
            message = f'{message} [skip ci]'

        LOGGER.info('amending commit with new message: %s', message)
        latest_tag = self.get_current_tag()

        if latest_tag:
            LOGGER.info('removing tag: %s', latest_tag)
            self.remove_tag(latest_tag)

        LOGGER.info('going back one commit')
        branch = self.repo.head.reference
        try:
            branch.commit = self.repo.head.commit.parents[0]
        except IndexError:
            LOGGER.error('cannot amend the first commit')
            sys.exit(-1)
        if files_to_add:
            self.stage_subset(*files_to_add)
        else:
            self.stage_all()
        self.repo.index.commit(message, skip_hooks=True)
        if latest_tag:
            LOGGER.info('resetting tag: %s', latest_tag)
            self.tag(latest_tag)