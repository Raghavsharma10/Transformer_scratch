def amend_commit(
            self,
            append_to_msg: typing.Optional[str] = None,
            new_message: typing.Optional[str] = None,
            files_to_add: typing.Optional[typing.Union[typing.List[str], str]] = None,
    ):
        """
        Amends last commit

        Args:
            append_to_msg: string to append to previous message
            new_message: new commit message
            files_to_add: optional list of files to commit
        """